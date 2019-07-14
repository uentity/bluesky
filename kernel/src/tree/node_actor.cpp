/// @file
/// @author uentity
/// @date 14.07.2019
/// @brief Implementataion of node actor
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/atoms.h>
#include <bs/kernel/config.h>
#include "node_actor.h"

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

auto node_actor::insert(sp_link L, const InsertPolicy pol) -> insert_status<Key::ID> {
	// can't move persistent node from it's owner
	if(!L || !accepts(L) || (L->flags() & Flags::Persistent && L->owner()))
		return {end<Key::ID>(), false};

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
						L->rename_silent(std::move(new_name));
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
			return {dup, is_inserted};
		}
	}
	// try to insert given link
	return I.insert(std::move(L));
}

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

NAMESPACE_END(blue_sky::tree)
