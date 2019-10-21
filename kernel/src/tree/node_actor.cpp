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

#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <caf/actor_ostream.hpp>
#include <caf/stateful_actor.hpp>
#include <caf/others.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

node_actor::node_actor(caf::actor_config& cfg, const std::string& id, timespan data_timeout)
	: super(cfg), timeout_(data_timeout)
{
	bind_new_id(id);
}

// implement shallow links copy ctor
node_actor::node_actor(caf::actor_config& cfg, const std::string& id, caf::actor src_a)
	: node_actor(cfg, id)
{
	auto src = caf::actor_cast<node_actor*>(src_a);
	if(!src) throw error{"Bad source passed to node_actor copy constructor"};

	timeout_ = src->timeout_;
	allowed_otypes_ = src->allowed_otypes_;
	for(const auto& plink : src->links_.get<Key_tag<Key::AnyOrder>>()) {
		// non-deep clone can result in unconditional moving nodes from source
		insert(plink->clone(true), InsertPolicy::AllowDupNames);
	}
	// [NOTE] links are in invalid state (no owner set) now
	// correct this by manually calling `node::propagate_owner()` after copy is constructed
}

auto node_actor::goodbye() -> void {
	if(self_grp) {
		send(self_grp, a_bye());
		leave(self_grp);
	}

	// unload retranslators from leafs
	disconnect();
}

auto node_actor::disconnect() -> void {
	//auto solo = std::lock_guard{ links_guard_ };
	//for(const auto& L : links_)
	//	stop_retranslate_from(L);

	auto& Reg = system().registry();
	for(auto& [lid, rs] : axons_) {
		for(auto& ractor : { Reg.get(rs.first), Reg.get(rs.second) })
			send(caf::actor_cast<caf::actor>(ractor), a_bye(), a_ack());
	}
	auto solo = std::lock_guard{ links_guard_ };
	axons_.clear();
}

auto node_actor::bind_new_id(const std::string& new_id) -> void {
	// leave old group
	if(self_grp) {
		if(new_id == self_grp.get()->identifier()) return;
		// rebind friends to new ID
		// [NOTE] don't send bye, otherwise retranslators from this will quit
		send(self_grp, a_bind_id(), new_id);
		leave(self_grp);
		//aout(this) << "node: rebind from " << self_grp.get()->identifier() <<
		//	" to " << new_id << std::endl;
	}
	//else {
	//	aout(this) << "node: join self group " << new_id << std::endl;
	//}

	// create self local group & join into it
	self_grp = system().groups().get_local(new_id);
	join(self_grp);

	// rebind retranslators to new group
	auto& Reg = system().registry();
	for(const auto& [lid, rsl_ids] : axons_) {
		for(auto& ractor : { Reg.get(rsl_ids.first), Reg.get(rsl_ids.second) })
			send(caf::actor_cast<caf::actor>(ractor), a_bind_id(), a_ack(), new_id); // a_ack to rebing target
	}
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
	// silently drop all other messages not in my character
	self->set_default_handler(caf::drop);

	// remember target node group
	self->state.tgt_grp = std::move(node_grp);
	// connect source
	self->state.src_grp = system().groups().get_local(to_string(lid));
	self->state.src_id = std::move(lid);
	self->join(self->state.src_grp);
	//aout(self) << "link retranslator started: " << self->state.src_grp_id() << " -> " <<
	//	self->state.tgt_grp_id() << std::endl;

	// register self
	const auto sid = self->id();
	system().registry().put(sid, self);

	return {
		// quit after source
		[=](a_bye, a_ack) {
			self->leave(self->state.src_grp);
			system().registry().erase(sid);
			//aout(self) << "link retranslator quit: " << to_string(self->state.src_id) <<
			//	" -> " << self->state.tgt_grp_id() << std::endl;
		},

		// rebind target group to new ID after node
		[=](a_bind_id, a_ack, const std::string& new_nid) {
			if(new_nid == self->state.tgt_grp_id() || new_nid == nil_oid) return;
			//aout(self) << "link retranslator rebound target: " << new_nid << std::endl;
			self->state.tgt_grp = system().groups().get_local(new_nid);
		},

		// retranslate events
		[=](a_lnk_rename, a_ack, const std::string& new_name, const std::string& old_name) {
			//aout(self) << "retranslate: rename: link " << to_string(self->state.src_id) <<
			//	" -> node " << self->state.tgt_grp_id() << std::endl;
			self->send(self->state.tgt_grp, a_lnk_rename(), a_ack(), self->state.src_id, new_name, old_name);
		},

		[=](a_lnk_status, a_ack, Req req, ReqStatus new_s, ReqStatus prev_s) {
			//aout(self) << "retranslate: status: link " << to_string(self->state.src_id) <<
			//	" -> node " << self->state.tgt_grp_id() << std::endl;
			self->send(self->state.tgt_grp, a_lnk_status(), a_ack(), self->state.src_id, req, new_s, prev_s);
		}
	};
}

// actor that retranslate some of link's messages attaching a link's ID to them
auto node_retranslator(caf::stateful_actor<node_rsl_state>* self, caf::group node_grp, std::string subnode_id)
-> caf::behavior {
	// remember target node group
	self->state.tgt_grp = std::move(node_grp);
	// connect source
	self->send(self, a_bind_id(), std::move(subnode_id));
	//self->state.src_grp = system().groups().get_local(subnode_id);
	//self->join(self->state.src_grp);
	//aout(self) << "node retranslator started: " << subnode_id << " -> " <<
	//	self->state.tgt_grp_id() << std::endl;

	// register self
	const auto sid = self->id();
	system().registry().put(sid, self);

	// silently drop all other messages not in my character
	self->set_default_handler([](caf::scheduled_actor* self, caf::message_view& mv) {
		return caf::drop(self, mv);
	});

	return {
		// follow source node and rebind source group after it
		[=](a_bind_id, const std::string& new_sid) {
			if(new_sid == self->state.src_grp_id() || new_sid == nil_oid) return;
			self->leave(self->state.src_grp);
			self->state.src_grp = system().groups().get_local(new_sid);
			self->join(self->state.src_grp);
			//aout(self) << "node retranslator (re)started: " << new_sid <<
			//	" -> " << self->state.tgt_grp_id() << std::endl;
		},

		// rebind target group to new ID after node
		[=](a_bind_id, a_ack, const std::string& new_tid) {
			if(new_tid == self->state.tgt_grp_id() || new_tid == nil_oid) return;
			//aout(self) << "node retranslator rebound target: " << new_tid << std::endl;
			self->state.tgt_grp = system().groups().get_local(new_tid);
		},

		// quit following source
		[=](a_bye, a_ack) {
			self->leave(self->state.src_grp);
			system().registry().erase(sid);
			//aout(self) << "node retranslator quit: " << self->state.src_grp_id() <<
			//	" -> " << self->state.tgt_grp_id() << std::endl;
		},

		// retranslate events
		[=](a_lnk_rename, a_ack, link::id_type lid, const std::string& new_name, const std::string& old_name) {
			//aout(self) << "retranslate: rename: node " << self->state.src_grp_id() <<
			//	" -> node " << self->state.tgt_grp_id() << std::endl;
			self->send(self->state.tgt_grp, a_lnk_rename(), a_ack(), lid, new_name, old_name);
		},

		[=](a_lnk_status, a_ack, link::id_type lid, Req req, ReqStatus new_s, ReqStatus prev_s) {
			//aout(self) << "retranslate: status: node " << self->state.src_grp_id() <<
			//	" -> node " << self->state.tgt_grp_id() << std::endl;
			self->send(self->state.tgt_grp, a_lnk_status(), a_ack(), lid, req, new_s, prev_s);
		},

		[=](a_lnk_insert, a_ack, link::id_type lid) {
			//aout(self) << "retranslate: insert: node " << self->state.src_grp_id() <<
			//	" -> node " << self->state.tgt_grp_id() << std::endl;
			self->send(self->state.tgt_grp, a_lnk_insert(), a_ack(), lid);
		},

		[=](a_lnk_erase, a_ack, std::vector<link::id_type> lids, std::vector<std::string> oids) {
			//aout(self) << "retranslate: erase: node " << self->state.src_grp_id() <<
			//	" -> node " << self->state.tgt_grp_id() << std::endl;
			self->send(self->state.tgt_grp, a_lnk_erase(), a_ack(), lids, oids);
		}
	};
}

NAMESPACE_END()

auto node_actor::retranslate_from(const sp_link& L) -> void {
	const auto& lid = L->id();
	axons_[lid] = {
		system().spawn(link_retranslator, self_grp, lid).id(),
		//-1
		system().spawn(node_retranslator, self_grp, L->oid()).id()
	};
	//caf::aout(this) << "*-* node: retranslating events from link " << L->name() << std::endl;
}

auto node_actor::stop_retranslate_from(const sp_link& L) -> void {
	auto prs = axons_.find(L->id());
	if(prs == axons_.end()) return;
	auto& rs_ids = prs->second;
	auto& Reg = system().registry();

	// stop link & subnode retranslators
	for(auto& ractor : { Reg.get(rs_ids.first), Reg.get(rs_ids.second) })
		send(caf::actor_cast<caf::actor>(ractor), a_bye(), a_ack());
	axons_.erase(L->id());
	//caf::aout(this) << "*-* node: stopped retranslating events from link " << L->name() << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
//  leafs insert & erase
//
auto node_actor::insert(sp_link L, const InsertPolicy pol, const sp_node& n) -> insert_status<Key::ID> {
	// can't move persistent node from it's owner
	if(!L || !accepts(L) || (L->flags() & Flags::Persistent && L->owner()))
		return { end<Key::ID>(), false };

	// make insertion in one single transaction
	auto solo = std::lock_guard{ links_guard_ };
	auto& Limpl = *L->pimpl();
	// [NOTE] shared lock is enough to prevent parallel modification
	// and required to allow 'reading' functions like `oid()` share a lock
	auto Lguard = std::shared_lock{ Limpl.guard_ };

	// check if we have duplicated name
	iterator<Key::ID> dup;
	if(enumval(pol & 3) > 0) {
		dup = find<Key::Name, Key::ID>(Limpl.name_);
		if(dup != end<Key::ID>() && (*dup)->id() != Limpl.id_) {
			bool unique_found = false;
			// first check if dup names are prohibited
			if(enumval(pol & InsertPolicy::DenyDupNames)) return {dup, false};
			else if(enumval(pol & InsertPolicy::RenameDup) && !(Limpl.flags_ & Flags::Persistent)) {
				// try to auto-rename link
				std::string new_name;
				for(int i = 0; i < 10000; ++i) {
					new_name = Limpl.name_ + '_' + std::to_string(i);
					if(find<Key::Name, Key::Name>(new_name) == end<Key::Name>()) {
						// we've found a unique name
						// [NOTE] rename is executed by actor in separate thread that will wait
						// until `Lguard` is released
						L->rename(std::move(new_name));
						unique_found = true;
						break;
					}
				}
			}
			// if no unique name was found - return fail
			if(!unique_found) return {dup, false};
		}
	}

	const auto make_result = [=](insert_status<Key::ID>&& res) {
		if(res.second) {
			const auto& child_L = *res.first;
			// create link exevents retranlator
			retranslate_from(child_L);
			// send message that link inserted
			send(self_grp, a_lnk_insert(), a_ack(), child_L->id());
		}
		return std::move(res);
	};

	// sym links can return proper OID only if owner is valid
	auto prev_owner = Limpl.owner_;
	if(n) Limpl.owner_ = n;
	// restore prev owner on exit
	auto finally = scope_guard{ [&]{
		Limpl.owner_ = prev_owner;
	}};

	// check for duplicating OID
	auto& I = links_.get<Key_tag<Key::ID>>();
	if( enumval(pol & (InsertPolicy::DenyDupOID | InsertPolicy::ReplaceDupOID)) ) {
		dup = find<Key::OID, Key::ID>(L->oid());
		if(dup != end<Key::ID>()) {
			bool is_inserted = false;
			if(enumval(pol & InsertPolicy::ReplaceDupOID))
				is_inserted = I.replace(dup, std::move(L));
			return make_result({ dup, is_inserted });
		}
	}
	// try to insert given link
	return make_result(I.insert(std::move(L)));
}

// postprocessing of just inserted link
// if link points to node, return it
auto node_actor::adjust_inserted_link(const sp_link& lnk, const sp_node& n) -> sp_node {
	// sanity
	if(!lnk) return nullptr;

	// change link's owner
	// [NOTE] for sym links it's important to set new owner early
	// otherwise, `data()` will return null and statuses will become Error
	auto prev_owner = lnk->owner();
	if(prev_owner != n) {
		if(prev_owner) prev_owner->erase(lnk->id());
		lnk->reset_owner(n);
	}

	// if we're inserting a node, relink it to ensure a single hard link exists
	return lnk->propagate_handle().value_or(nullptr);
}

auto node_actor::erase_impl(iterator<Key::ID> victim) -> void {
	if(victim == end<Key::ID>()) return;
	auto solo = std::lock_guard{ links_guard_ };

	auto& L = *victim;
	stop_retranslate_from(L);

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
	send(self_grp, a_lnk_erase(), a_ack(), lids, oids);
	// and erase link
	links_.get<Key_tag<Key::ID>>().erase(victim);
}

///////////////////////////////////////////////////////////////////////////////
//  misc
//
auto node_actor::accepts(const sp_link& what) const -> bool {
	if(!allowed_otypes_.size()) return true;
	const auto& what_type = what->obj_type_id();
	for(const auto& otype : allowed_otypes_) {
		if(what_type == otype) return true;
	}
	return false;
}

auto node_actor::set_handle(const sp_link& new_handle) -> void {
	// remove node from existing owner if it differs from owner of new handle
	if(const auto old_handle = handle_.lock()) {
		const auto owner = old_handle->owner();
		if(owner && (!new_handle || owner != new_handle->owner()))
			owner->erase(old_handle->id());
	}

	// set new handle link
	handle_ = new_handle;
}

void node_actor::on_rename(const Key_type<Key::ID>& key) {
	auto solo = std::lock_guard{ links_guard_ };

	// find target link by it's ID
	auto& I = links_.get<Key_tag<Key::ID>>();
	auto pos = I.find(key);
	// invoke replace as most safe & easy choice
	if(pos != I.end())
		I.replace(pos, *pos);
}

///////////////////////////////////////////////////////////////////////////////
//  behavior
//
auto node_actor::make_behavior() -> behavior_type { return {
	/// skip `id bind` & `bye` (should always come from myself)
	[](a_bind_id, const std::string&) {},
	[=](a_bye) {},

	// handle link rename
	[=](a_lnk_rename, a_ack, link::id_type lid, const std::string&, const std::string&) {
		on_rename(lid);
	},

	[=](a_lnk_status, a_ack, const link::id_type& lid, Req, ReqStatus new_s, ReqStatus) {
		// rebind source of subnode retranslator
		if(new_s == ReqStatus::OK) {
			if(auto L = find<Key::ID>(lid); L != links_.end()) {
				auto subn_rsl = system().registry().get(axons_[lid].second);
				// no `a_ack` to rebind source
				send(caf::actor_cast<caf::actor>(subn_rsl), a_bind_id(), (*L)->oid());
				//caf::aout(this) << "try to rebind node retranslator..." << std::endl;
			}
		}
	},

	// [TODO] add impl
	[=](a_lnk_insert, a_ack, link::id_type lid) {},
	// [TODO] add impl
	[=](a_lnk_erase, a_ack, std::vector<link::id_type>, std::vector<std::string>) {}
}; }

NAMESPACE_END(blue_sky::tree)
