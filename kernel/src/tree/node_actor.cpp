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
#include "link_impl.h"

#include <bs/log.h>
#include <bs/tree/tree.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <caf/typed_event_based_actor.hpp>

#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)
using namespace allow_enumops;
using namespace kernel::radio;
using namespace std::chrono_literals;

[[maybe_unused]] auto adbg_impl(node_actor* A) -> caf::actor_ostream {
	auto res = caf::aout(A);
	res << "[N] ";
	if(auto pgrp = A->home(unsafe).get())
		res << "[" << pgrp->identifier() << "]";
	else
		res << "[homeless]";
	return res <<  ": ";
}

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

	set_default_handler([](auto*, auto&) -> caf::result<caf::message> {
		return caf::none;
	});
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

///////////////////////////////////////////////////////////////////////////////
//  leafs insert & erase
//
auto node_actor::insert(
	link L, const InsertPolicy pol, bool silent
) -> insert_status<Key::ID> {
	adbg(this) << "{a_lnk_insert}" << (silent ? " silent: " : ": ") <<
		to_string(L.id()) << std::endl;

	return impl.insert(std::move(L), pol, [=](const link& child_L) {
		// send message that link inserted (with position)
		if(!silent) send<high_prio>(
			home(unsafe), a_ack(), this, a_node_insert(),
			child_L.id(), impl.index(impl.find<Key::ID>(child_L.id())), pol
		);
	});
}

auto node_actor::insert(
	link L, std::size_t to_idx, const InsertPolicy pol, bool silent
) -> node::insert_status {
	// 1. insert an element using ID index
	// [NOTE] silent insert - send ack message later
	auto res = insert(std::move(L), pol, true);
	auto res_idx = impl.index<Key::ID>(res.first);
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
				home(unsafe), a_ack(), this, a_node_insert(), std::move(lid), to_idx, pol
			);
		else if(to != from) // move
			send<high_prio>(
				home(unsafe), a_ack(), this, a_node_insert(), std::move(lid), to_idx, *res_idx
			);
	}
	return { to_idx, res.second };
}


NAMESPACE_BEGIN()

auto on_erase(const link& L, node_actor& self) {
	// collect link IDs of all deleted subtree elements
	// first elem is erased link itself
	lids_v lids{ L.id() };
	walk(L, [&lids](const link&, std::list<link> Ns, std::vector<link> Os) {
		const auto dump_erased = [&](const link& erl) {
			lids.push_back(erl.id());
		};
		std::for_each(Ns.cbegin(), Ns.cend(), dump_erased);
	});

	// send message that link erased
	self.send<high_prio>(
		self.home(unsafe), a_ack(), &self, a_node_erase(), std::move(lids)
	);
}

NAMESPACE_END()

auto node_actor::erase(const lid_type& victim, EraseOpts opts) -> size_t {
	const auto ppf = [=](const link& L) { on_erase(L, *this); };
	std::size_t res = 0;
	error::eval_safe([&] { res = impl.erase<Key::ID>(
		victim,
		enumval(opts & EraseOpts::Silent) ? noop : function_view{ ppf },
		enumval(opts & EraseOpts::DontResetOwner)
	); });
	return res;
}

///////////////////////////////////////////////////////////////////////////////
//  primary behavior
//
auto node_actor::make_primary_behavior() -> primary_actor_type::behavior_type {
return {
	// unconditionally join home group - used after deserialization
	[=](a_hi) { join(home()); },

	[=](a_home) { return impl.home_; },

	// skip `bye` (should always come from myself)
	[=](a_bye) {},

	[=](a_node_gid) -> std::string { return gid(); },

	// get handle
	[=](a_node_handle) { return link{ impl.handle() }; },

	// set handle
	[=](a_node_handle, link new_handle) {
		// remove node from existing owner if it differs from owner of new handle
		if(const auto old_handle = impl.handle()) {
			const auto owner = old_handle.owner();
			if(owner && (!new_handle || owner != new_handle.owner()))
				owner->erase(old_handle.id());
		}

		if(new_handle)
			impl.handle_ = new_handle;
		else
			impl.handle_.reset();
	},

	// get size
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

	[=](a_node_keys, Key order) -> caf::result<lids_v> {
		// builtin indexes can be processed directly
		switch(order) {
		case Key::ID : return impl.keys<Key::ID>();
		case Key::AnyOrder : return impl.keys<Key::ID, Key::AnyOrder>();
		case Key::Name : return impl.keys<Key::ID, Key::Name>();
		default: break;
		}
		// others via extra index actor
		return delegate(
			system().spawn(extraidx_search_actor),
			a_node_keys(), order, impl.leafs(Key::AnyOrder)
		);
	},

	[=](a_node_ikeys, Key order) -> caf::result<std::vector<std::size_t>> {
		// builtin indexes can be processed directly
		switch(order) {
		case Key::ID : return impl.keys<Key::AnyOrder, Key::ID>();
		case Key::AnyOrder : return impl.keys<Key::AnyOrder>();
		case Key::Name : return impl.keys<Key::AnyOrder, Key::Name>();
		default: break;
		}

		// others via extra sorted leafs
		auto rp = make_response_promise();
		request(
			system().spawn(extraidx_search_actor), def_timeout(true),
			a_node_leafs(), order, impl.leafs(Key::AnyOrder)
		).then(
			[=](const links_v& leafs) mutable {
				rp.deliver( impl.ikeys(leafs.begin(), leafs.end()) );
			}
		);
		return rp;
	},

	[=](a_node_keys, Key meaning, Key order) -> caf::result<std::vector<std::string>> {
		return delegate(
			system().spawn(extraidx_search_actor),
			a_node_keys(), meaning, order, impl.leafs(Key::AnyOrder)
		);
	},

	// find
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

	// insert new link
	[=](a_node_insert, link L, InsertPolicy pol) -> node::insert_status {
		adbg(this) << "{a_node_insert}" << std::endl;
		auto res = insert(std::move(L), pol);
		return { impl.index<Key::ID>(std::move(res.first)), res.second };
	},

	// insert into specified position
	[=](a_node_insert, link L, std::size_t idx, InsertPolicy pol) -> node::insert_status {
		adbg(this) << "{a_node_insert}" << std::endl;
		return insert(std::move(L), idx, pol);
	},

	// insert bunch of links
	[=](a_node_insert, links_v Ls, InsertPolicy pol) {
		size_t cnt = 0;
		for(auto& L : Ls) {
			if(insert(std::move(L), pol).second) ++cnt;
		}
		return cnt;
	},

	// normal link erase
	[=](a_node_erase, const lid_type& lid) -> std::size_t {
		return erase(lid);
	},

	// all other erase overloads do normal erase
	[=](a_node_erase, std::size_t idx) {
		return impl.erase<Key::AnyOrder>(
			idx, [=](const link& L) { on_erase(L, *this); }
		);
	},

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

	// erase bunch of links
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

	// apply custom order
	[=](a_node_rearrange, const std::vector<std::size_t>& new_order) -> error::box {
		return impl.rearrange<Key::AnyOrder>(new_order);
	},

	[=](a_node_rearrange, const lids_v& new_order) -> error::box {
		return impl.rearrange<Key::ID>(new_order);
	},

	/// Non-public extensions
	// erase link by ID with specified options
	[=](a_node_erase, const lid_type& lid, EraseOpts opts) -> std::size_t {
		return erase(lid, opts);
	},
}; }

auto node_actor::make_behavior() -> behavior_type {
	return actor_type::behavior_type{first_then_second(
		make_ack_behavior(), make_primary_behavior()
	)}.unbox();
}

NAMESPACE_END(blue_sky::tree)
