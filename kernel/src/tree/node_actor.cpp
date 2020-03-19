/// @file
/// @author uentity
/// @date 14.07.2019
/// @brief Implementataion of node actor
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_actor.h"
#include "node_extraidx_actor.h"
#include "node_retranslators.h"
#include "link_impl.h"
#include <bs/log.h>
#include <bs/tree/tree.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <caf/typed_event_based_actor.hpp>

#include <cereal/types/optional.hpp>

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)
using namespace allow_enumops;
using namespace kernel::radio;
using namespace std::chrono_literals;

#if DEBUG_ACTOR == 1

static auto adbg(node_actor* A) -> caf::actor_ostream {
	auto res = caf::aout(A);
	res << "[N] ";
	if(auto pgrp = A->home(unsafe).get())
		res << "[" << pgrp->identifier() << "]";
	else
		res << "[null grp]";
	return res <<  ": ";
}

#endif

/*-----------------------------------------------------------------------------
 *  node_actor
 *-----------------------------------------------------------------------------*/
node_actor::node_actor(caf::actor_config& cfg, sp_nimpl Nimpl)
	: super(cfg), pimpl_(std::move(Nimpl)), impl([this]() -> node_impl& {
		if(!pimpl_) throw error{"node actor: bad (null) node impl passed"};
		return *pimpl_;
	}())
{
	// prevent termination in case some errors happens in group members
	// for ex. if they receive unexpected messages (translators normally do)
	set_error_handler([this](caf::error er) {
		switch(static_cast<caf::sec>(er.code())) {
		case caf::sec::unexpected_message :
			break;
		default:
			default_error_handler(this, er);
		}
	});

	set_default_handler(caf::drop);
}

node_actor::~node_actor() = default;

auto node_actor::on_exit() -> void {
	adbg(this) << "dies" << std::endl;

	// be polite with everyone
	goodbye();
	// [IMPORTANT] manually reset pimpl, otherwise cycle won't break:
	// actor dtor never called until at least one strong ref to it still exists
	// (even though behavior is terminated by sending `exit` message)
	pimpl_.reset();
}

auto node_actor::goodbye() -> void {
	adbg(this) << "goodbye" << std::endl;
	if(auto& H = home(unsafe)) {
		send(H, a_bye());
		leave(H);
	}

	// unload retranslators from leafs
	disconnect();
}

auto node_actor::name() const -> const char* {
	return "node_actor";
}

auto node_actor::home() -> caf::group& {
	return impl.home_ ? impl.home_ : home({});
}
auto node_actor::home(unsafe_t) const -> caf::group& {
	return impl.home_;
}

auto node_actor::home(std::string gid) -> caf::group& {
	// [IMPORTANT] pass silent = true to prevent feedback
	join(impl.home( std::move(gid), true ));
	return impl.home_;
}

auto node_actor::gid() -> const std::string& {
	return home().get()->identifier();
}
auto node_actor::gid(unsafe_t) const -> std::string {
	return impl.home_ ? impl.home_.get()->identifier() : "";
}

auto node_actor::disconnect() -> void {
	//for(const auto& L : links_)
	//	stop_retranslate_from(L);

	auto& Reg = system().registry();
	for(auto& [lid, rs] : axons_) {
		// stop link retranslator
		send<high_prio>(caf::actor_cast<caf::actor>(Reg.get(rs.first)), a_bye());
		// and subnode
		if(rs.second)
			send<high_prio>(caf::actor_cast<caf::actor>(Reg.get(*rs.second)), a_bye());
	}
	axons_.clear();
}

auto node_actor::retranslate_from(const link& L) -> void {
	const auto lid = L.id();
	auto& AS = system();

	// spawn link retranslator first
	auto axon = axon_t{
		AS.spawn(link_retranslator, home(), lid).id(), {}
	};
	// for hard links also listen to subtree
	// [TODO] move this part to `link::propagate_handle()`
	if(L.type_id() == hard_link::type_id_())
		axon.second = AS.spawn(node_retranslator, home(), lid, link::actor(L)).id();

	axons_[lid] = std::move(axon);
	//adbg(this) << "*-* node: retranslating events from link " << L->name() << std::endl;
}

auto node_actor::stop_retranslate_from(const link& L) -> void {
	auto prs = axons_.find(L.id());
	if(prs == axons_.end()) return;
	auto& rs = prs->second;
	auto& Reg = system().registry();

	// stop link retranslator
	send<high_prio>(caf::actor_cast<caf::actor>(Reg.get(rs.first)), a_bye());
	// .. and for subnode
	if(rs.second)
		send<high_prio>(caf::actor_cast<caf::actor>(Reg.get(*rs.second)), a_bye());
	axons_.erase(prs);
	//adbg(this) << "*-* node: stopped retranslating events from link " << L->name() << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
//  leafs insert & erase
//
auto node_actor::insert(
	link L, const InsertPolicy pol, bool silent
) -> insert_status<Key::ID> {
	adbg(this) << "{a_lnk_insert}" << (silent ? " silent: " : ": ") <<
		to_string(L.id()) << std::endl;

	return impl.insert(std::move(L), pol, [=](const link& child_L) {
		// create link events retranlator
		retranslate_from(child_L);
		// send message that link inserted (with position)
		if(!silent) send<high_prio>(
			home(unsafe), a_ack(), a_node_insert(),
			child_L.id(), impl.to_index(impl.find<Key::ID>(child_L.id())), pol
		);
	});
}

auto node_actor::insert(
	link L, std::size_t to_idx, const InsertPolicy pol, bool silent
) -> node::insert_status {
	// 1. insert an element using ID index
	// [NOTE] silent insert - send ack message later
	auto res = insert(std::move(L), pol, true);
	auto res_idx = impl.to_index<Key::ID>(res.first);
	if(!res_idx) return { res_idx, res.second };

	// 2. reposition an element in AnyOrder index
	to_idx = std::min(to_idx, impl.size());
	auto from = impl.project<Key::ID>(res.first);
	auto to = std::next(impl.begin(), to_idx);
	// noop if to == from
	pimpl_->links_.get<Key_tag<Key::AnyOrder>>().relocate(to, from);

	// detect move and send proper message
	auto lid = (*res.first).id();
	if(!silent) {
		if(res.second) // normal insert
			send<high_prio>(
				home(unsafe), a_ack(), a_node_insert(), std::move(lid), to_idx, pol
			);
		else if(to != from) // move
			send<high_prio>(
				home(unsafe), a_ack(), a_node_insert(), std::move(lid), to_idx, *res_idx
			);
	}
	return { to_idx, res.second };
}


NAMESPACE_BEGIN()

auto on_erase(const link& L, node_actor& self) {
	self.stop_retranslate_from(L);

	// collect link IDs of all deleted subtree elements
	// first elem is erased link itself
	lids_v lids{ L.id() };
	std::vector<std::string> oids{ L.oid() };
	walk(L, [&lids, &oids](const link&, std::list<link> Ns, std::vector<link> Os) {
		const auto dump_erased = [&](const link& erl) {
			lids.push_back(erl.id());
			oids.push_back(erl.oid());
		};
		std::for_each(Ns.cbegin(), Ns.cend(), dump_erased);
		std::for_each(Os.cbegin(), Os.cend(), dump_erased);
	});

	// send message that link erased
	self.send<high_prio>(
		self.home(unsafe), a_ack(), a_node_erase(), std::move(lids), std::move(oids)
	);
}

NAMESPACE_END()

auto node_actor::erase(const lid_type& victim, EraseOpts opts) -> size_t {
	const auto ppf = [=](const link& L) { on_erase(L, *this); };
	return impl.erase<Key::ID>(
		victim,
		enumval(opts & EraseOpts::Silent) ? noop_postproc_f : function_view{ ppf },
		bool(enumval(opts & EraseOpts::DontResetOwner))
	);
}

///////////////////////////////////////////////////////////////////////////////
//  behavior
//
auto node_actor::make_behavior() -> behavior_type {
	using typed_behavior = typename node_impl::actor_type::behavior_type;

	return typed_behavior{
		// unconditionally join home group - used after deserialization
		[=](a_hi) { join(home()); },

		// skip `bye` (should always come from myself)
		[=](a_bye) {},

		[=](a_node_gid) -> std::string { return gid(); },

		[=](a_node_disconnect) { disconnect(); },

		// 2. propagate owner
		//[=](a_node_propagate_owner, bool deep) { impl.propagate_owner(deep); },

		// 3. get handle
		[=](a_node_handle) { return link{ impl.handle() }; },

		// 4. get size
		[=](a_node_size) { return impl.size(); },

		[=](a_node_leafs, Key order) -> caf::result<links_v> {
			adbg(this) << "{a_node_leafs} " << static_cast<int>(order) << std::endl;
			if(has_builtin_index(order))
				return impl.leafs(order);
			else
				return delegate(
					system().spawn(extraidx_search_actor),
					a_node_leafs(), order, impl.leafs(Key::AnyOrder)
				);
		},

		// 5.
		[=](a_node_find, const lid_type& lid) -> link {
			adbg(this) << "{a_node_find LID} " << to_string(lid) << std::endl;
			auto res = impl.search<Key::ID>(lid);
			return res;
		},

		[=](a_node_find, std::size_t idx) -> link {
			adbg(this) << "{a_node_find idx} " << idx << std::endl;
			return impl.search<Key::AnyOrder>(idx);
		},

		[=](a_node_find, std::string key, Key key_meaning) -> caf::result<link> {
			adbg(this) << "{a_node_find key} " << key << std::endl;
			if(has_builtin_index(key_meaning)) {
				auto res = link{};
				error::eval_safe([&]{ res = impl.search(key, key_meaning); });
				return res;
			}
			else
				return delegate(
					system().spawn(extraidx_search_actor),
					a_node_find(), std::move(key), key_meaning, impl.values<Key::AnyOrder>()
				);
		},

		// deep search
		[=](a_node_deep_search, lid_type lid) -> caf::result<link> {
			adbg(this) << "{a_node_deep_search}" << std::endl;
			return delegate(
				system().spawn(extraidx_deep_search_actor, handle()),
				a_node_deep_search(), std::move(lid)
			);
		},

		[=](a_node_deep_search, std::string key, Key key_meaning, bool search_all)
		-> caf::result<links_v> {
			adbg(this) << "{a_node_deep_search}" << std::endl;
			return delegate(
				system().spawn(extraidx_deep_search_actor, handle()),
				a_node_deep_search(), std::move(key), key_meaning, search_all
			);
		},

		// index
		[=](a_node_index, const lid_type& lid) -> existing_index {
			return impl.index<Key::ID>(lid);
		},

		[=](a_node_index, std::string key, Key key_meaning) -> caf::result<existing_index> {
			if(has_builtin_index(key_meaning)) {
				auto res = existing_index{};
				error::eval_safe([&] { res = impl.index(key, key_meaning); });
				return res;
			}
			else
				return delegate(
					system().spawn(extraidx_search_actor),
					a_node_index(), std::move(key), key_meaning, impl.values<Key::AnyOrder>()
				);
		},

		// equal_range
		[=](a_node_equal_range, std::string key, Key key_meaning) -> caf::result<links_v> {
			if(has_builtin_index(key_meaning)) {
				auto res = links_v{};
				error::eval_safe([&] { res = impl.equal_range(key, key_meaning); });
				return res;
			}
			else
				return delegate(
					system().spawn(extraidx_search_actor),
					a_node_equal_range(), std::move(key), key_meaning, impl.values<Key::AnyOrder>()
				);
		},

		// 8. insert new link
		[=](a_node_insert, link L, InsertPolicy pol) -> node::insert_status {
			adbg(this) << "{a_node_insert}" << std::endl;
			auto res = insert(std::move(L), pol);
			return { impl.to_index<Key::ID>(std::move(res.first)), res.second };
		},

		// 9. insert into specified position
		[=](a_node_insert, link L, std::size_t idx, InsertPolicy pol) -> node::insert_status {
			adbg(this) << "{a_node_insert}" << std::endl;
			return insert(std::move(L), idx, pol);
		},

		// 10. insert bunch of links
		[=](a_node_insert, links_v Ls, InsertPolicy pol) {
			size_t cnt = 0;
			for(auto& L : Ls) {
				if(insert(std::move(L), pol).second) ++cnt;
			}
			return cnt;
		},

		// 13. normal link erase
		[=](a_node_erase, const lid_type& lid) -> std::size_t {
			return erase(lid);
		},

		// 14. all other erase overloads do normal erase
		[=](a_node_erase, std::size_t idx) {
			return impl.erase<Key::AnyOrder>(
				idx, [=](const link& L) { on_erase(L, *this); }
			);
		},

		// 15.
		[=](a_node_erase, std::string key, Key key_meaning) -> caf::result<std::size_t>{
			if(has_builtin_index(key_meaning)) {
				auto res = std::size_t{};
				error::eval_safe([&] {
					res = impl.erase(
						key, key_meaning, [=](const link& L) { on_erase(L, *this); }
					);
				});
				return res;
			}
			else
				return delegate(
					system().spawn(extraidx_erase_actor, handle()),
					a_node_erase(), std::move(key), key_meaning, impl.values<Key::AnyOrder>()
				);
		},

		// 16. erase bunch of links
		[=](a_node_erase, const lids_v& lids) {
			return impl.erase(
				lids, [=](const link& L) { on_erase(L, *this); }
			);
		},

		[=](a_node_clear) {
			impl.links_.clear();
		},

		// rename
		[=](a_lnk_rename, const lid_type& lid, const std::string& new_name) -> std::size_t {
			return impl.rename<Key::ID>(lid, new_name);
		},

		[=](a_lnk_rename, std::size_t idx, const std::string& new_name) -> std::size_t {
			return impl.rename<Key::AnyOrder>(idx, new_name);
		},

		[=](a_lnk_rename, const std::string& old_name, const std::string& new_name) -> std::size_t {
			return impl.rename<Key::Name>(old_name, new_name);
		},

		// 18. apply custom order
		[=](a_node_rearrange, const std::vector<std::size_t>& new_order) {
			impl.rearrange<Key::AnyOrder>(new_order);
		},

		[=](a_node_rearrange, const lids_v& new_order) {
			impl.rearrange<Key::ID>(new_order);
		},

		/// Non-public extensions
		// 13. erase link by ID with specified options
		[=](a_node_erase, const lid_type& lid, EraseOpts opts) -> std::size_t {
			return erase(lid, opts);
		},

		// 11. ack on insert - reflect insert from sibling node actor
		[=](a_ack, a_node_insert, lid_type lid, size_t pos, InsertPolicy pol) {
			adbg(this) << "{a_node_insert ack}" << std::endl;
			if(auto S = current_sender(); S != this) {
				request(caf::actor_cast<caf::actor>(S), impl.timeout, a_node_find(), std::move(lid))
				.then([=](link L) {
					// [NOTE] silent insert
					insert(std::move(L), pos, pol, true);
				});
			}
		},
		// 12. ack on move
		[=](a_ack, a_node_insert, lid_type lid, size_t to, size_t from) {
			if(auto S = current_sender(); S != this) {
				if(auto p = impl.find<Key::ID, Key::ID>(lid); p != impl.end<Key::ID>()) {
					insert(*p, to, InsertPolicy::AllowDupNames, true);
				}
			}
		},

		// 17. ack on erase - reflect erase from sibling node actor
		[=](a_ack, a_node_erase, const lids_v& lids, const std::vector<std::string>&) {
			if(auto S = current_sender(); S != this && !lids.empty()) {
				erase(lids.front(), EraseOpts::Silent);
			}
		},

		// 6. handle link rename
		[=](a_ack, a_lnk_rename, lid_type lid, const std::string&, const std::string&) {
			adbg(this) << "{a_lnk_rename}" << std::endl;
			impl.refresh(lid);
		},

		// 7. track link status
		[=](a_ack, a_lnk_status, const lid_type& lid, Req req, ReqStatus new_s, ReqStatus) {
			// refresh link if new data arrived
			if(new_s == ReqStatus::OK) {
				impl.refresh(lid);
			}
		},

	}.unbox();
}

NAMESPACE_END(blue_sky::tree)
