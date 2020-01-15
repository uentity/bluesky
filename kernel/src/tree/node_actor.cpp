/// @file
/// @author uentity
/// @date 14.07.2019
/// @brief Implementataion of node actor
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_actor.h"
#include "link_impl.h"
#include <bs/log.h>
#include <bs/tree/tree.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

//#include <caf/actor_ostream.hpp>
#include <caf/stateful_actor.hpp>
#include <caf/others.hpp>

#include <cereal/types/optional.hpp>

#define DEBUG_ACTOR 0

#if DEBUG_ACTOR == 1
#include <caf/actor_ostream.hpp>
#endif

OMIT_ITERATORS_SERIALIZATION
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<blue_sky::tree::node::id_type>)

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;
using namespace std::chrono_literals;

NAMESPACE_BEGIN()
#if DEBUG_ACTOR == 1

template<typename Actor>
auto adbg(Actor* A, const std::string& nid = {}) -> caf::actor_ostream {
	auto res = caf::aout(A);
	if constexpr(std::is_same_v<Actor, node_actor>) {
		res << "[N] ";
		if(auto pgrp = A->impl.self_grp.get())
			res << "[" << A->impl.self_grp.get()->identifier() << "]";
		else
			res << "[null grp]";
		res <<  ": ";
	}
	else if(!nid.empty() && nid.front() != '"') {
		res << "[N] [" << nid << "]: ";
	}
	return res;
}

#else

template<typename Actor>
constexpr auto adbg(const Actor*, const std::string& = {}) { return log::D(); }

#endif
NAMESPACE_END()

static boost::uuids::string_generator uuid_from_str;

/*-----------------------------------------------------------------------------
 *  node_actor
 *-----------------------------------------------------------------------------*/
node_actor::node_actor(caf::actor_config& cfg, caf::group ngrp, sp_nimpl Nimpl)
	: super(cfg), pimpl_(std::move(Nimpl)), impl([this]() -> node_impl& {
		if(!pimpl_) throw error{"node actor: bad (null) node impl passed"};
		return *pimpl_;
	}())
{
	// remember link's local group
	impl.self_grp = std::move(ngrp);

	// on exit say goodbye to self group
	set_exit_handler([this](caf::exit_msg& er) {
		goodbye();
		default_exit_handler(this, er);
	});

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

auto node_actor::goodbye() -> void {
	adbg(this) << "goodbye" << std::endl;
	if(impl.self_grp) {
		send(impl.self_grp, a_bye());
		leave(impl.self_grp);
	}

	// unload retranslators from leafs
	disconnect();
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

auto node_actor::name() const -> const char* {
	return "node_actor";
}

///////////////////////////////////////////////////////////////////////////////
//  leafs events retranslators
//
NAMESPACE_BEGIN()
// state for node & link retranslators
struct node_rsl_state {
	caf::group src_grp;
	caf::group tgt_grp;

	auto src_grp_id() const -> const std::string& {
		return src_grp ? src_grp.get()->identifier() : nil_grp_id;
	}
	auto tgt_grp_id() const -> const std::string& {
		return tgt_grp ? tgt_grp.get()->identifier() : nil_grp_id;
	}
};

struct link_rsl_state : node_rsl_state {
	link::id_type src_id;
};

// actor that retranslate some of link's messages attaching a link's ID to them
auto link_retranslator(caf::stateful_actor<link_rsl_state>* self, caf::group node_grp, link::id_type lid)
-> caf::behavior {
	// remember target node group
	self->state.tgt_grp = std::move(node_grp);
	// connect source
	//auto lid = L->id();
	self->state.src_grp = system().groups().get_local(to_string(lid));
	self->state.src_id = std::move(lid);
	self->join(self->state.src_grp);

	auto sdbg = [=](const std::string& msg_name = {}) {
		auto res = adbg(self, self->state.tgt_grp_id()) << "<- [L] [" << self->state.src_grp_id() << "] ";
		//auto res = caf::aout(self) << self->state.tgt_grp_id() << " <- ";
		if(!msg_name.empty())
			res << '{' << msg_name << "} ";
		return res;
	};
	sdbg() << "retranslator started" << std::endl;

	// register self
	const auto sid = self->id();
	system().registry().put(sid, self);

	// silently drop all other messages not in my character
	self->set_default_handler(caf::drop);

	return {
		// quit after source
		[=](a_bye) {
			self->leave(self->state.src_grp);
			system().registry().erase(sid);
			sdbg() << "retranslator quit" << std::endl;
		},

		// retranslate events
		[=](a_ack, a_lnk_rename, std::string new_name, std::string old_name) {
			sdbg("rename") << old_name << " -> " << new_name << std::endl;
			self->send<high_prio>(
				self->state.tgt_grp, a_ack(), a_lnk_rename(), self->state.src_id,
				std::move(new_name), std::move(old_name)
			);
		},

		[=](a_ack, a_lnk_status, Req req, ReqStatus new_s, ReqStatus prev_s) {
			sdbg("status") << to_string(req) << ": " << to_string(prev_s) << " -> " << to_string(new_s);
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_lnk_status(), self->state.src_id, req, new_s, prev_s);
		}
	};
}

// actor that retranslate some of link's messages attaching a link's ID to them
auto node_retranslator(caf::stateful_actor<node_rsl_state>* self, caf::group node_grp, std::string subnode_id)
-> caf::behavior {
	// remember target node group
	self->state.tgt_grp = std::move(node_grp);
	// join subnode group
	self->state.src_grp = system().groups().get_local(subnode_id);
	self->join(self->state.src_grp);

	auto sdbg = [=](const std::string& msg_name = {}) {
		auto res = adbg(self, self->state.tgt_grp_id()) << "<- [N] [" << self->state.src_grp_id() << "] ";
		//auto res = caf::aout(self) << self->state.tgt_grp_id() << " <- ";
		if(!msg_name.empty())
			res << '{' << msg_name << "} ";
		return res;
	};
	sdbg() << "retranslator started" << std::endl;

	// register self
	const auto sid = self->id();
	system().registry().put(sid, self);

	// silently drop all other messages not in my character
	self->set_default_handler(caf::drop);

	return {
		// quit following source
		[=](a_bye) {
			self->leave(self->state.src_grp);
			system().registry().erase(sid);
			sdbg() << "retranslator quit" << std::endl;
		},

		// retranslate events
		[=](a_ack, a_lnk_rename, link::id_type lid, std::string new_name, std::string old_name) {
			sdbg("rename") << old_name << " -> " << new_name << std::endl;
			self->send<high_prio>(
				self->state.tgt_grp, a_ack(), a_lnk_rename(), lid, std::move(new_name), std::move(old_name)
			);
		},

		[=](a_ack, a_lnk_status, link::id_type lid, Req req, ReqStatus new_s, ReqStatus prev_s) {
			sdbg("status") << to_string(req) << ": " << to_string(prev_s) << " -> " << to_string(new_s);
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_lnk_status(), lid, req, new_s, prev_s);
		},

		[=](a_ack, a_lnk_insert, link::id_type lid, std::size_t pos, InsertPolicy pol) {
			sdbg("insert") << to_string(lid) << " in pos " << pos << std::endl;
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_lnk_insert(), lid, pos, pol);
		},
		[=](a_ack, a_lnk_insert, link::id_type lid, std::size_t to_idx, std::size_t from_idx) {
			sdbg("insert move") << to_string(lid) << " pos " << from_idx << " -> " << to_idx << std::endl;
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_lnk_insert(), lid, to_idx, from_idx);
		},

		[=](a_ack, a_lnk_erase, std::vector<link::id_type> lids, std::vector<std::string> oids) {
			sdbg("erase") << (lids.empty() ? "" : to_string(lids[0])) << std::endl;
			self->send<high_prio>(self->state.tgt_grp, a_ack(), a_lnk_erase(), lids, oids);
		}
	};
}

NAMESPACE_END()

auto node_actor::retranslate_from(const sp_link& L) -> void {
	const auto& lid = L->id();
	auto& AS = system();

	// spawn link retranslator first
	auto axon = axon_t{ AS.spawn(link_retranslator, impl.self_grp, lid).id(), {} };
	// and node (if pointee is a node)
	L->data_node_gid().map([&](std::string gid) {
		axon.second = AS.spawn(node_retranslator, impl.self_grp, std::move(gid)).id();
	});
	axons_[lid] = std::move(axon);
	//adbg(this) << "*-* node: retranslating events from link " << L->name() << std::endl;
}

auto node_actor::stop_retranslate_from(const sp_link& L) -> void {
	auto prs = axons_.find(L->id());
	if(prs == axons_.end()) return;
	auto& rs = prs->second;
	auto& Reg = system().registry();

	// stop link retranslator
	send<high_prio>(caf::actor_cast<caf::actor>(Reg.get(rs.first)), a_bye());
	// and subnode
	if(rs.second)
		send<high_prio>(caf::actor_cast<caf::actor>(Reg.get(*rs.second)), a_bye());
	axons_.erase(prs);
	//adbg(this) << "*-* node: stopped retranslating events from link " << L->name() << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
//  leafs insert & erase
//
auto node_actor::insert(
	sp_link L, const InsertPolicy pol, bool silent
) -> insert_status<Key::ID> {
	adbg(this) << "{a_lnk_insert}" << (silent ? " silent: " : ": ") <<
		to_string(L->id()) << std::endl;

	return impl.insert(std::move(L), pol, [=](const sp_link& child_L) {
		// create link exevents retranlator
		retranslate_from(child_L);
		// send message that link inserted (with position)
		if(!silent) send<high_prio>(
			impl.self_grp, a_ack(), a_lnk_insert(),
			child_L->id(), (size_t)std::distance(impl.begin(), impl.find<Key::ID>(child_L->id())), pol
		);
	});
}

auto node_actor::insert(
	sp_link L, std::size_t to_idx, const InsertPolicy pol, bool silent
) -> std::pair<size_t, bool> {
	// 1. insert an element using ID index
	// [NOTE] silent insert, send message later below
	auto res = insert(std::move(L), pol, true);

	// 2. reposition an element in AnyOrder index
	if(res.first != impl.end<Key::ID>()) {
		auto from = impl.project<Key::ID>(res.first);
		to_idx = std::min(to_idx, impl.size());
		auto to = std::next(impl.begin(), to_idx);
		if(to != from)
			pimpl_->links_.get<Key_tag<Key::AnyOrder>>().relocate(to, from);

		// detect move and send proper message
		auto lid = (*res.first)->id();
		if(!silent) {
			if(res.second) // normal insert
				send<high_prio>(
					impl.self_grp, a_ack(), a_lnk_insert(), std::move(lid), to_idx, pol
				);
			else if(to != from) // move
				send<high_prio>(
					impl.self_grp, a_ack(), a_lnk_insert(), std::move(lid), to_idx,
					(size_t)std::distance(impl.begin(), from)
				);
		}
		return { std::distance(impl.begin(), to), res.second };
	}
	return { impl.size(), res.second };
}


NAMESPACE_BEGIN()

auto on_erase(const sp_link& L, node_actor& self) {
	self.stop_retranslate_from(L);

	// collect link IDs & obj IDs of all deleted subtree elements
	// first elem is erased link itself
	std::vector<link::id_type> lids{ L->id() };
	std::vector<std::string> oids{ L->oid() };
	walk(L, [&lids, &oids](const sp_link&, std::list<sp_link> Ns, std::vector<sp_link> Os) {
		const auto dump_erased = [&](const sp_link& erl) {
			lids.push_back(erl->id());
			oids.push_back(erl->oid());
		};
		std::for_each(Ns.cbegin(), Ns.cend(), dump_erased);
		std::for_each(Os.cbegin(), Os.cend(), dump_erased);
	});

	// send message that link erased
	self.send<high_prio>(self.impl.self_grp, a_ack(), a_lnk_erase(), lids, oids);
}

NAMESPACE_END()

auto node_actor::erase(const link::id_type& victim, EraseOpts opts) -> size_t {
	const auto ppf = [=](const sp_link& L) { on_erase(L, *this); };
	return impl.erase<Key::ID>(
		victim,
		enumval(opts & EraseOpts::Silent) ? noop_postproc_f : function_view{ ppf },
		bool(enumval(opts & EraseOpts::DontResetOwner))
	);
}

auto node_actor::erase(std::size_t idx) -> size_t {
	return impl.erase<Key::AnyOrder>(
		idx, [=](const sp_link& L) { on_erase(L, *this); }
	);
}

auto node_actor::erase(const std::string& key, Key key_meaning) -> size_t {
	const auto ppf = [=](const sp_link& L) { on_erase(L, *this); };
	switch(key_meaning) {
	case Key::ID:
		return impl.erase<Key::ID>(uuid_from_str(key), ppf);
	case Key::OID:
		return impl.erase<Key::OID>(key, ppf);
	case Key::Type:
		return impl.erase<Key::Type>(key, ppf);
	default:
	case Key::Name:
		return impl.erase<Key::Name>(key, ppf);
	}
}

///////////////////////////////////////////////////////////////////////////////
//  behavior
//
auto node_actor::make_behavior() -> behavior_type {
	using typed_behavior = typename node_impl::actor_type::behavior_type;

	return typed_behavior{
		// 0.
		[=](a_node_gid) -> std::string {
			return impl.gid();
		},

		// 0, skip `bye` (should always come from myself)
		[=](a_bye) {},


		// 2. propagate owner
		[=](a_node_propagate_owner, bool deep) { impl.propagate_owner(deep); },

		// 3. get handle
		[=](a_node_handle) { return impl.handle_.lock(); },

		// 4. get size
		[=](a_node_size) { return impl.size(); },

		[=](a_node_leafs, Key order) { return impl.leafs(order); },

		// 5.
		[=](a_lnk_find, const link::id_type& lid) -> sp_link {
			if(auto p = impl.find<Key::ID, Key::ID>(lid); p != impl.end<Key::ID>())
				return *p;
			return nullptr;
		},

		// 6. handle link rename
		[=](a_ack, a_lnk_rename, link::id_type lid, const std::string&, const std::string&) {
			adbg(this) << "{a_lnk_rename}" << std::endl;
			impl.refresh(lid);
		},

		// 7. track link status
		[=](a_ack, a_lnk_status, const link::id_type& lid, Req req, ReqStatus new_s, ReqStatus) {
			// refresh link if new data arrived
			if(new_s == ReqStatus::OK) {
				impl.refresh(lid);
			}
		},

		// 8. insert new link
		[=](a_lnk_insert, sp_link L, InsertPolicy pol) -> node::actor_insert_status {
			adbg(this) << "{a_lnk_insert}" << std::endl;
			auto res = insert(std::move(L), pol);
			return {
				res.first != impl.end<Key::ID>() ?
					std::distance(impl.begin<Key::AnyOrder>(), impl.project<Key::ID>(res.first)) :
					std::optional<size_t>{},
				res.second
			};
		},

		// 9. insert into specified position
		[=](a_lnk_insert, sp_link L, std::size_t idx, InsertPolicy pol) -> node::actor_insert_status {
			adbg(this) << "{a_lnk_insert}" << std::endl;
			return insert(std::move(L), idx, pol);
		},

		// 10. insert bunch of links
		[=](a_lnk_insert, std::vector<sp_link> Ls, InsertPolicy pol) {
			size_t cnt = 0;
			for(auto& L : Ls) {
				if(insert(std::move(L), pol).second) ++cnt;
			}
			return cnt;
		},

		// 11. ack on insert - reflect insert from sibling node actor
		[=](a_ack, a_lnk_insert, link::id_type lid, size_t pos, InsertPolicy pol) {
			adbg(this) << "{a_lnk_insert ack}" << std::endl;
			if(auto S = current_sender(); S != this) {
				request(caf::actor_cast<caf::actor>(S), impl.timeout, a_lnk_find(), std::move(lid))
				.then([=](sp_link L) {
					// [NOTE] silent insert
					insert(std::move(L), pos, pol, true);
				});
			}
		},
		// 12. ack on move
		[=](a_ack, a_lnk_insert, link::id_type lid, size_t to, size_t from) {
			if(auto S = current_sender(); S != this) {
				if(auto p = impl.find<Key::ID, Key::ID>(lid); p != impl.end<Key::ID>()) {
					insert(*p, to, InsertPolicy::AllowDupNames, true);
				}
			}
		},

		// 13. erase link by ID with specified options
		[=](a_lnk_erase, const link::id_type& lid, EraseOpts opts) -> std::size_t {
			return erase(lid, opts);
		},
		// 14. all other erase overloads do normal erase
		[=](a_lnk_erase, std::size_t idx) {
			return erase(idx);
		},
		// 15.
		[=](a_lnk_erase, const std::string& key, Key key_meaning) {
			return erase(key, key_meaning);
		},
		// 16. erase bunch of links
		[=](a_lnk_erase, const std::vector<link::id_type>& lids) {
			size_t cnt = 0;
			for(const auto& lid : lids)
				cnt += erase(lid);
			return cnt;
		},

		// 17. ack on erase - reflect erase from sibling node actor
		[=](a_ack, a_lnk_erase, const std::vector<link::id_type>& lids, const std::vector<std::string>&) {
			if(auto S = current_sender(); S != this && !lids.empty()) {
				erase(lids.front(), EraseOpts::Silent);
			}
		},

		// 18. apply custom order
		[=](a_node_rearrange, const std::vector<std::size_t>& new_order) {
			impl.rearrange<Key::AnyOrder>(new_order);
		},
		// 19.
		[=](a_node_rearrange, const std::vector<node::id_type>& new_order) {
			impl.rearrange<Key::ID>(new_order);
		},
	}.unbox();
}

NAMESPACE_END(blue_sky::tree)
