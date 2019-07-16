/// @file
/// @author uentity
/// @date 14.07.2019
/// @brief Implementataion of node actor
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_actor.h"

#include <boost/uuid/uuid_io.hpp>
#include <caf/actor_ostream.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

node_actor::node_actor(caf::actor_config& cfg, const std::string& id, timespan data_timeout)
	: super(cfg), timeout_(data_timeout)
{
	// create self local group & join into it
	self_grp = kernel::config::actor_system().groups().get_local(id);
	join(self_grp);
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
	send(self_grp, a_bye());
	leave(self_grp);
}

///////////////////////////////////////////////////////////////////////////////
//  leafs events retranslators
//
NAMESPACE_BEGIN()

// actor that retranslate some of link's messages attaching a link's ID to them
auto link_retranslator(caf::event_based_actor* self, caf::group node_grp, link::id_type lid) -> caf::behavior {
	// join link's group
	auto Lgrp = actor_system().groups().get_local( to_string(lid) );
	self->join(Lgrp);
	// register self
	const auto sid = self->id();
	actor_system().registry().put(sid, self);

	// silently drop all other messages not in my character
	self->set_default_handler([](caf::scheduled_actor* self, caf::message_view& mv) {
		return caf::drop(self, mv);
	});

	return {
		// quit after link
		[=](a_bye) {
			self->leave(Lgrp);
			kernel::config::actor_system().registry().erase(sid);
		},

		// retranslate events
		[=](a_lnk_rename, a_ack, const std::string& new_name, const std::string& old_name) {
			self->send(node_grp, a_lnk_rename(), a_ack(), lid, new_name, old_name);
		},

		[=](a_lnk_status, a_ack, Req req, ReqStatus new_s, ReqStatus prev_s) {
			self->send(node_grp, a_lnk_status(), a_ack(), lid, req, new_s, prev_s);
		}
	};
}

// actor that retranslate some of link's messages attaching a link's ID to them
auto node_retranslator(caf::event_based_actor* self, caf::group node_grp, const std::string& subnode_id) -> caf::behavior {
	// join link's group
	auto Sgrp = actor_system().groups().get_local( subnode_id );
	self->join(Sgrp);
	// register self
	const auto sid = self->id();
	actor_system().registry().put(sid, self);

	// silently drop all other messages not in my character
	self->set_default_handler([](caf::scheduled_actor* self, caf::message_view& mv) {
		return caf::drop(self, mv);
	});

	return {
		// quit after link
		[=](a_bye) {
			self->leave(Sgrp);
			kernel::config::actor_system().registry().erase(sid);
		},

		// retranslate events
		[=](a_lnk_rename, a_ack, link::id_type lid, const std::string& new_name, const std::string& old_name) {
			self->send(node_grp, a_lnk_rename(), a_ack(), lid, new_name, old_name);
		},

		[=](a_lnk_status, a_ack, link::id_type lid, Req req, ReqStatus new_s, ReqStatus prev_s) {
			self->send(node_grp, a_lnk_status(), a_ack(), lid, req, new_s, prev_s);
		},

		[=](a_lnk_insert, a_ack, link::id_type lid) {
			self->send(node_grp, a_lnk_insert(), a_ack(), lid);
		},

		[=](a_lnk_erase, a_ack, link::id_type lid) {
			self->send(node_grp, a_lnk_erase(), a_ack(), lid);
		}
	};
}

NAMESPACE_END()

///////////////////////////////////////////////////////////////////////////////
//  leafs insert & erase
//
auto node_actor::insert(sp_link L, const InsertPolicy pol) -> insert_status<Key::ID> {
	const auto make_result = [=](insert_status<Key::ID>&& res) {
		if(res.second) {
			// create retranlators for inserted link & subnode (if any)
			const auto& child_L = **res.first;
			const auto& lid = child_L.id();
			axons_[lid] = {
				actor_system().spawn(link_retranslator, self_grp, lid).id(),
				actor_system().spawn(node_retranslator, self_grp, L->oid()).id()
			};

			// send message that link inserted
			send(self_grp, a_lnk_insert(), a_ack(), lid);
		}
		return std::move(res);
	};

	// can't move persistent node from it's owner
	if(!L || !accepts(L) || (L->flags() & Flags::Persistent && L->owner()))
		return { end<Key::ID>(), false };

	// make insertion in one single transaction
	auto solo = std::lock_guard{ links_guard_ };
	// check if we have duplication name
	iterator<Key::ID> dup;
	if(enumval(pol & 3) > 0) {
		dup = find<Key::Name, Key::ID>(L->name());
		if(dup != end<Key::ID>() && (*dup)->id() != L->id()) {
			bool unique_found = false;
			// first check if dup names are prohibited
			if(enumval(pol & InsertPolicy::DenyDupNames)) return {dup, false};
			else if(enumval(pol & InsertPolicy::RenameDup) && !(L->flags() & Flags::Persistent)) {
				// try to auto-rename link
				std::string new_name;
				for(int i = 0; i < 10000; ++i) {
					new_name = L->name() + '_' + std::to_string(i);
					if(find<Key::Name, Key::Name>(new_name) == end<Key::Name>()) {
						// we've found a unique name
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

	const auto lid = (*victim)->id();
	const auto rs = axons_[lid];
	auto& Reg = actor_system().registry();

	// stop link & subnode retranslators
	for(auto& ractor : { Reg.get(rs.first), Reg.get(rs.second) })
		send(caf::actor_cast<caf::actor>(ractor), a_bye());
	axons_.erase(lid);

	// send message that link erased
	send(self_grp, a_lnk_erase(), a_ack(), lid);
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
	// bye comes from myself
	[=](a_bye) {
		// suspend link leafs retranslators
		auto solo = std::lock_guard{ links_guard_ };
		auto& Reg = actor_system().registry();
		for(auto& [lid, rs] : axons_) {
			for(auto& ractor : { Reg.get(rs.first), Reg.get(rs.second) })
				send(caf::actor_cast<caf::actor>(ractor), a_bye());
		}
		axons_.clear();
	},

	// handle link rename
	[=](a_lnk_rename, a_ack, link::id_type lid, const std::string&, const std::string&) {
		on_rename(lid);
	},
	// skip link status - not interested
	[](a_lnk_status, a_ack, link::id_type, Req, ReqStatus, ReqStatus) {},

	// [TODO] add impl
	[=](a_lnk_insert, a_ack, link::id_type lid) {},
	// [TODO] add impl
	[=](a_lnk_erase, a_ack, link::id_type lid) {}
}; }

NAMESPACE_END(blue_sky::tree)
