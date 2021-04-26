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

#include <algorithm>

NAMESPACE_BEGIN(blue_sky::tree)
using bs_detail::shared;
	
ENGINE_TYPE_DEF(node_impl, "node")

auto node_impl::clone(node_actor* papa, bool deep) const -> caf::result<sp_nimpl> {
	auto res_promise = papa->make_response_promise<sp_nimpl>();
	auto do_clone = [=](auto&& self, auto work, auto res) mutable {
		if(work.empty()) {
			res_promise.deliver(std::move(res));
			return;
		}

		auto leaf = work.back();
		work.pop_back();
		// [NOTE] use `await` to ensure node is not modified while cloning leafs
		papa->request(leaf.actor(), kernel::radio::timeout(true), a_clone{}, deep)
		.await(
			[=, work = std::move(work)](const link& leaf_clone) mutable {
				auto& res_leafs = res->links_.template get<Key_tag<Key::AnyOrder>>();
				res_leafs.insert(res_leafs.end(), leaf_clone);
				self(self, std::move(work), std::move(res));
			},
			// stop process on error & return existing result
			[=](const caf::error& er) mutable {
				forward_caf_error(er, "in node_impl::clone()").dump();
				res_promise.deliver(std::move(res));
			}
		);
	};

	auto work = leafs(Key::AnyOrder);
	std::reverse(work.begin(), work.end());
	do_clone(do_clone, std::move(work), std::make_shared<node_impl>());
	return res_promise;
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
			caf::anon_send(
				node_impl::actor(owner), a_node_erase(), old_handle.id(), EraseOpts::Normal
			);
	}

	if(new_handle)
		handle_ = new_handle;
	else
		handle_.reset();
}

auto node_impl::propagate_owner(const node& super, bool deep) -> void {
	// setup super
	if(super == nil_node::nil_engine()) return;
	reset_super_engine(super);

	// correct owner of single link & return a node it points to
	const auto adjust_link = [&](const link& lnk) -> decltype(auto) {
		// change link's owner
		if(auto prev_owner = lnk.owner(); prev_owner != super) {
			lnk.pimpl()->reset_owner(super);
			if(prev_owner)
				caf::anon_send(
					node_impl::actor(prev_owner), a_node_erase(), lnk.id(), EraseOpts::Normal
				);
		}

		// relink node to ensure it' handle is correct
		return lnk.pimpl()->propagate_handle();
	};

	// iterate over children & correct their owners
	for(auto& L : links_) {
		auto child_node = adjust_link(L);
		if(deep)
			child_node.map([](auto& child_node) {
				child_node.pimpl()->propagate_owner(child_node, true);
			});
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

auto node_impl::keys(Key order) const -> lids_v {
	switch(order) {
	case Key::ID: return keys<Key::ID>();
	case Key::AnyOrder: return keys<Key::ID, Key::AnyOrder>();
	case Key::Name: return keys<Key::ID, Key::Name>();
	default: return {};
	}
}

auto node_impl::ikeys(Key order) const -> std::vector<std::size_t> {
	switch(order) {
	case Key::ID: return keys<Key::AnyOrder, Key::ID>();
	case Key::AnyOrder: return keys<Key::AnyOrder>();
	case Key::Name: return keys<Key::AnyOrder, Key::Name>();
	default: return {};
	}
}

auto node_impl::leafs(Key order) const -> links_v {
	switch(order) {
	case Key::AnyOrder: return values<Key::AnyOrder>();
	case Key::ID: return values<Key::ID>();
	case Key::Name: return values<Key::Name>();
	default: return {};
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

auto node_impl::rename(iterator<Key::Name> pos, std::string new_name) -> void {
	// rename & update index
	// must be called atomically for both node & link at given pos
	links_.get<Key_tag<Key::Name>>().modify(
		std::move(pos), [&](link& L) { L.pimpl()->rename(std::move(new_name)); }
	);
}

///////////////////////////////////////////////////////////////////////////////
//  leafs insert & erase
//
// hardcode number of rename trials on insertion
inline constexpr auto rename_trials = 10000;

// [NOTE] assume insertion happens atomically for link being inserted
auto node_impl::insert(link L, InsertPolicy pol) -> insert_status<Key::ID> {
	using namespace allow_enumops;

	using R = insert_status<Key::ID>;
	const auto inserr = R{ links_.get<Key_tag<Key::ID>>().end(), false };
	if(!L) return inserr;
	auto& Limpl = *L.pimpl();
	const auto Lflags = Limpl.flags_;
	const auto prev_owner = L.owner();

	// can't move persistent node from it's owner
	if((Lflags & Flags::Persistent) && prev_owner)
		return inserr;

	// 1. check if we have duplicated name and have to rename link after insertion
	// If insert policy deny duplicating name & we have to rename link being inserted
	// then new link name will go here
	auto Lname = std::optional<std::string>{};
	if(enumval(pol & 3) > 0) {
		const auto& old_name = Limpl.name_;
		auto dup = find<Key::Name, Key::ID>(old_name);
		if(dup != end<Key::ID>() && dup->id() != Limpl.id_) {
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
	auto res = links_.get<Key_tag<Key::ID>>().insert(L);

	// 3. postprocess
	if(res.second) {
		// remove from prev parent and propagate handle
		if(super_ != prev_owner) {
			if(prev_owner)
				// erase won't touch owner (we set it manually)
				caf::anon_send<high_prio>(
					node_impl::actor(prev_owner), a_node_erase(), Limpl.id_, EraseOpts::Normal
				);
			// set owner to this node
			Limpl.reset_owner(super_engine());
			// ensure stored nodes (if any) has single handle `L`
			Limpl.propagate_handle();
		}

		// rename link if needed
		if(Lname)
			rename(project<Key::ID, Key::Name>(res.first), std::move(*Lname));
	}
	else {
		// check if we need to deep merge given links
		// go one step down the hierarchy
		if(enumval(pol & InsertPolicy::Merge) && res.first != end<Key::ID>()) {
			auto& res_L = *res.first;
			auto src_node = L.data_node();
			auto dst_node = res_L.data_node();
			if(src_node && dst_node) {
				// insert all links from source node into destination
				caf::anon_send(dst_node.actor(), a_node_insert(), src_node.leafs(), pol);
			}
		}
	}
	return res;
}

auto node_impl::erase_impl(iterator<Key::ID> victim, leaf_postproc_fn ppf) -> std::size_t {
	// preprocess before erasing
	auto L = *victim;
	ppf(L);

	// reset link's owner inly if it's owner matches self
	if(auto Limpl = L.pimpl(); Limpl->owner_ == super_)
		Limpl->reset_owner(node::nil());
	// erase
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

NAMESPACE_END(blue_sky::tree)
