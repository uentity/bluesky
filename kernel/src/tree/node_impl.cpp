/// @file
/// @author uentity
/// @date 14.07.2019
/// @brief Implementataion of node actor
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_impl.h"
#include "link_impl.h"
#include "node_actor.h"
#include "nil_engine.h"

#include <bs/log.h>
#include <bs/tree/tree.h>
#include <bs/detail/tuple_utils.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

NAMESPACE_BEGIN(blue_sky::tree)
using bs_detail::shared;
	
ENGINE_TYPE_DEF(node_impl, "node")

node_impl::node_impl(const links_v& leafs) :
	links_(leafs.begin(), leafs.end())
{
	// [NOTE] links are in invalid state (no owner set) now
	// correct this by manually calling `node::propagate_owner()` after copy is constructed
}

auto node_impl::clone(bool deep) const -> sp_nimpl {
	auto res = std::make_unique<node_impl>();
	auto& res_leafs = res->links_.get<Key_tag<Key::AnyOrder>>();
	for(const auto& leaf : links_.get<Key_tag<Key::AnyOrder>>())
		res_leafs.insert(res_leafs.end(), leaf.clone(deep));
	return res;
}

auto node_impl::handle() const -> link {
	auto guard = lock(shared);
	return handle_.lock();
}

auto node_impl::set_handle(const link& new_handle) -> void {
	auto guard = lock();
	// remove node from existing owner if it differs from owner of new handle
	// [NOTE] don't call `handle()` here, because mutex is already locked
	if(const auto old_handle = handle_.lock()) {
		const auto owner = old_handle.owner();
		if(owner && (!new_handle || owner != new_handle.owner()))
			owner.erase(old_handle.id());
	}

	if(new_handle)
		handle_ = new_handle;
	else
		handle_.reset();
}

// postprocessing of just inserted link
// if link points to node, return it
auto node_impl::adjust_inserted_link(const link& lnk, const node& target) -> node {
	// sanity
	if(!lnk) return node::nil();

	// change link's owner
	if(auto prev_owner = lnk.owner(); prev_owner != target) {
		lnk.pimpl()->reset_owner(target);
		// [NOTE] instruct prev node to not reset link's owner - we set above
		if(prev_owner)
			caf::anon_send(
				node_impl::actor(prev_owner), a_node_erase(), lnk.id(), EraseOpts::DontResetOwner
			);
	}

	// if we're inserting a node, relink it to ensure a single hard link exists
	return lnk.pimpl()->propagate_handle().value_or(node::nil());
}

auto node_impl::propagate_owner(const engine& S, bool deep) -> void {
	// setup super
	if(S == nil_node::nil_engine()) return;
	reset_super_engine(S);
	// properly setup owner in node's leafs
	for(auto& L : links_) {
		auto child_node = node_impl::adjust_inserted_link(L, S);
		if(deep && child_node)
			child_node.pimpl()->propagate_owner(child_node, true);
	}
}

auto node_impl::spawn_actor(sp_nimpl nimpl) -> caf::actor {
	// if home group is already started (for ex. by deserializer), then use it
	// otherwise obtain new group with UUID
	auto ngrp = nimpl->home ? nimpl->home : system().groups().get_local(to_string(gen_uuid()));
	return spawn_nactor(std::move(nimpl), std::move(ngrp));
}

auto node_impl::size() const -> std::size_t {
	return links_.size();
}

auto node_impl::leafs(Key order) const -> links_v {
	switch(order) {
	case Key::AnyOrder:
		return values<Key::AnyOrder>();
	case Key::ID:
		return values<Key::ID>();
	case Key::Name:
		return values<Key::Name>();
	default:
		return {};
	}
}

auto node_impl::search(const std::string& key, Key key_meaning) const -> link {
	switch(key_meaning) {
	case Key::ID:
		return to_uuid(key).map([&](lid_type lid) { return search<Key::ID>(lid); }).value_or(link{});
	case Key::Name:
		return search<Key::Name>(key);
	default:
		return {};
	}
}

auto node_impl::index(const std::string& key, Key key_meaning) const -> existing_index {
	switch(key_meaning) {
	case Key::ID:
		return to_uuid(key).map([&](lid_type lid) { return index<Key::ID>(lid); }).value_or(existing_index{});
	case Key::Name:
		return index<Key::Name>(key);
	default:
		return {};
	}
}

auto node_impl::equal_range(const std::string& key, Key key_meaning) const -> links_v {
	switch(key_meaning) {
	case Key::ID:
		return to_uuid(key).map(
			[&](lid_type lid) { return equal_range<Key::ID>(lid).extract_values(); }
		).value_or(links_v{});
	case Key::Name:
		return equal_range<Key::Name>(key).extract_values();
	default:
		return {};
	}
}

///////////////////////////////////////////////////////////////////////////////
//  leafs insert & erase
//
// hardcode number of rename trials on insertion
inline constexpr auto rename_trials = 10000;

auto node_impl::insert(link L, const InsertPolicy pol) -> insert_status<Key::ID> {
	using namespace allow_enumops;

	// can't move persistent node from it's owner
	const auto Lflags = L.flags();
	if(!L || (Lflags & Flags::Persistent && L.owner()))
		return { {}, false };

	// 1. check if we have duplicated name and have to rename link after insertion
	// If insert policy deny duplicating name & we have to rename link being inserted
	// then new link name will go here
	auto Lname = std::optional<std::string>{};
	if(enumval(pol & 3) > 0) {
		const auto old_name = L.name();
		auto dup = find<Key::Name, Key::ID>(old_name);
		if(dup != end<Key::ID>() && dup->id() != L.id()) {
			// first check if dup names are prohibited
			if(enumval(pol & InsertPolicy::DenyDupNames) || Lflags & Flags::Persistent)
				return { std::move(dup), false };
			else if(enumval(pol & InsertPolicy::RenameDup)) {
				// try to auto-rename link
				auto names_end = end<Key::Name>();
				for(int i = 0; i < rename_trials; ++i) {
					auto new_name = old_name + '_' + std::to_string(i);
					if(find<Key::Name, Key::Name>(new_name) == names_end) {
						Lname = std::move(new_name);
						break;
					}
				}
			}
			// if no unique name was found - return fail
			if(!Lname) return { std::move(dup), false };
		}
	}

	// 2. make insertion
	// [NOTE] reset link's owner to insert safely (block symlink side effects, etc)
	const auto prev_owner = L.owner();
	L.pimpl()->reset_owner(node::nil());
	auto res = links_.get<Key_tag<Key::ID>>().insert(L);

	// 3. postprocess
	auto& res_L = *res.first;
	auto& res_L_impl = *res_L.pimpl();
	if(res.second) {
		// remove from prev parent and propagate handle while link's owner still NULL (important!)
		const auto self = super_engine();
		if(prev_owner && prev_owner != self)
			// erase won't touch owner (we set it manually)
			caf::anon_send(
				node_impl::actor(prev_owner), a_node_erase(), res_L.id(), EraseOpts::DontResetOwner
			);
		res_L_impl.propagate_handle();
		// set owner to this node
		res_L_impl.reset_owner(self);

		// rename link if needed
		if(Lname) res_L.rename(std::move(*Lname));
	}
	else {
		// restore link's original owner
		L.pimpl()->reset_owner(prev_owner);
		// check if we need to deep merge given links
		// go one step down the hierarchy
		if(enumval(pol & InsertPolicy::Merge) && res.first != end<Key::ID>()) {
			auto src_node = L.data_node();
			auto dst_node = res_L.data_node();
			if(src_node && dst_node) {
				// insert all links from source node into destination
				dst_node.insert(src_node.leafs(), pol);
			}
		}
	}
	return res;
}

auto node_impl::erase_impl(
	iterator<Key::ID> victim, leaf_postproc_fn ppf, bool dont_reset_owner
) -> std::size_t {
	// preprocess before erasing
	auto L = *victim;
	ppf(L);

	// erase
	if(!dont_reset_owner) L.pimpl()->reset_owner(node::nil());
	auto res = index<Key::ID>(victim);
	links_.get<Key_tag<Key::ID>>().erase(victim);
	return res.value_or(0);
}

auto node_impl::erase(const std::string& key, Key key_meaning, leaf_postproc_fn ppf) -> size_t {
	switch(key_meaning) {
	case Key::ID:
		return to_uuid(key).map([&](lid_type lid) { return erase<Key::ID>(lid, ppf); }).value_or(0);
	case Key::Name:
		return erase<Key::Name>(key, ppf);
	default:
		return 0;
	}
}

auto node_impl::erase(const lids_v& r, leaf_postproc_fn ppf) -> std::size_t {
	std::size_t res = 0;
	std::for_each(
		r.begin(), r.end(),
		[&](const auto& lid) { res += erase(lid, ppf); }
	);
	return res;
}

///////////////////////////////////////////////////////////////////////////////
//  misc
//
auto node_impl::refresh(const lid_type& lid) -> void {
	// find target link by it's ID
	auto& I = links_.get<Key_tag<Key::ID>>();
	if(auto pos = I.find(lid); pos != I.end())
		links_.get<Key_tag<Key::Name>>().modify_key( project<Key::ID, Key::Name>(pos), noop );
}

NAMESPACE_END(blue_sky::tree)
