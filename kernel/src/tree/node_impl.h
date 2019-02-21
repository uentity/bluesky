/// @file
/// @author uentity
/// @date 29.06.2018
/// @brief BS tree node implementation part of PIMPL
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <bs/tree/node.h>
#include <set>
#include <mutex>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

using links_container = node::links_container;
using Key = node::Key;
template<Key K> using iterator = typename node::iterator<K>;
template<Key K> using Key_tag = typename node::Key_tag<K>;
template<Key K> using Key_type = typename node::Key_type<K>;
template<Key K> using Key_const = typename node::Key_const<K>;
template<Key K> using insert_status = typename node::insert_status<K>;
template<Key K> using range = typename node::range<K>;

using Flags = link::Flags;
using Req = link::Req;
using ReqStatus = link::ReqStatus;

/*-----------------------------------------------------------------------------
 *  node_impl
 *-----------------------------------------------------------------------------*/
class node::node_impl {
public:
	friend struct access_node_impl;

	template<Key K = Key::ID>
	static sp_link deep_search_impl(
		const node_impl& n, const Key_type<K>& key,
		std::set<Key_type<Key::ID>> active_symlinks = {}
		//std::unordered_set<Key_type<Key::ID>, boost::hash<Key_type<Key::ID>>> active_symlinks = {}
	) {
		// first do direct search in leafs
		auto r = n.find<K, K>(key);
		if(r != n.end<K>()) return *r;

		// if not succeeded search in children nodes
		for(const auto& l : n.links_) {
			// remember symlink
			const auto is_symlink = l->type_id() == "sym_link";
			if(is_symlink){
				if(active_symlinks.find(l->id()) == active_symlinks.end())
					active_symlinks.insert(l->id());
				else continue;
			}
			// check populated status before moving to next level
			if(l->flags() & link::LazyLoad && l->req_status(Req::DataNode) != ReqStatus::OK)
				continue;
			// search on next level
			if(const auto next_n = l->data_node()) {
				auto next_l = deep_search_impl<K>(*next_n->pimpl_, key, active_symlinks);
				if(next_l) return next_l;
			}
			// remove symlink
			if(is_symlink)
				active_symlinks.erase(l->id());
		}
		return nullptr;
	}

	//static deep_merge(const node_impl& n, )

	template<Key K = Key::AnyOrder>
	auto begin() const {
		return links_.get<Key_tag<K>>().begin();
	}
	template<Key K = Key::AnyOrder>
	auto end() const {
		return links_.get<Key_tag<K>>().end();
	}

	template<Key K, Key R = Key::AnyOrder>
	auto find(
		const Key_type<K>& key,
		std::enable_if_t<std::is_same<Key_const<K>, Key_const<R>>::value>* = nullptr
	) const {
		return links_.get<Key_tag<K>>().find(key);
	}

	template<Key K, Key R = Key::AnyOrder>
	auto find(
		const Key_type<K>& key,
		std::enable_if_t<!std::is_same<Key_const<K>, Key_const<R>>::value>* = nullptr
	) const {
		return links_.project<Key_tag<R>>(
			links_.get<Key_tag<K>>().find(key)
		);
	}

	template<Key K = Key::ID>
	auto equal_range(const Key_type<K>& key) const {
		return links_.get<Key_tag<K>>().equal_range(key);
	}

	template<Key K = Key::ID>
	void erase(const Key_type<K>& key) {
		links_locker_t my_turn(links_guard_);
		links_.get<Key_tag<K>>().erase(key);
	}

	template<Key K = Key::ID>
	void erase(const range<K>& r) {
		links_locker_t my_turn(links_guard_);
		links_.get<Key_tag<K>>().erase(r.first, r.second);
	}

	template<Key K = Key::ID>
	sp_link deep_search(const Key_type<K>& key) const {
		return this->deep_search_impl<K>(*this, key);
	}

	insert_status<Key::ID> insert(sp_link L, const InsertPolicy pol) {
		// can't move persistent node from it's owner
		if(!L || !accepts(L) || (L->flags() & Flags::Persistent && L->owner()))
			return {end<Key::ID>(), false};

		// make insertion in one single transaction
		links_locker_t my_turn(links_guard_);
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

	template<Key K>
	std::vector<Key_type<K>> keys() const {
		std::set<Key_type<K>> r;
		auto kex = Key_tag<K>();
		for(const auto& i : links_)
			r.insert(kex(*i));
		return {r.begin(), r.end()};
	}

	template<Key K>
	bool rename(iterator<K>&& pos, std::string&& new_name) {
		links_locker_t my_turn(links_guard_);

		if(pos == end<K>()) return false;
		return links_.get<Key_tag<K>>().modify(pos, [name = std::move(new_name)](sp_link& l) {
			l->rename_silent(std::move(name));
		});
	}

	template<Key K>
	std::size_t rename(const Key_type<K>& key, const std::string& new_name, bool all = false) {
		links_locker_t my_turn(links_guard_);

		range<K> matched_items = equal_range<K>(key);
		auto& storage = links_.get<Key_tag<K>>();
		auto renamer = [&new_name](sp_link& l) {
			l->rename_silent(new_name);
		};
		int cnt = 0;
		for(auto pos = matched_items.begin(); pos != matched_items.end(); ++pos) {
			storage.modify(pos, renamer);
			++cnt;
			if(!all) break;
		}
		return cnt;
	}

	void on_rename(const Key_type<Key::ID>& key) {
		links_locker_t my_turn(links_guard_);

		// find target link by it's ID
		auto& I = links_.get<Key_tag<Key::ID>>();
		auto pos = I.find(key);
		// invoke replace as most safe & easy choice
		if(pos != I.end())
			I.replace(pos, *pos);
	}

	template<Key K>
	auto project(iterator<K> pos) const {
		return links_.project<Key_tag<Key::AnyOrder>>(std::move(pos));
	}

	bool accepts(const sp_link& what) const {
		if(!allowed_otypes_.size()) return true;
		const auto& what_type = what->obj_type_id();
		for(const auto& otype : allowed_otypes_) {
			if(what_type == otype) return true;
		}
		return false;
	}

	// implement shallow links copy ctor
	node_impl(const node_impl& src)
		: allowed_otypes_(src.allowed_otypes_)
	{
		for(const auto& plink : src.links_.get<Key_tag<Key::AnyOrder>>()) {
			// non-deep clone can result in unconditional moving nodes from source
			insert(plink->clone(true), InsertPolicy::AllowDupNames);
		}
		// [NOTE] links are in invalid state (no owner set) now
		// correct this by manually calling `node::propagate_owner()` after copy is constructed
	}

	void set_handle(const sp_link& new_handle) {
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
	static sp_node adjust_inserted_link(const sp_link& lnk, const sp_node& n) {
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

	node_impl() = default;

	std::weak_ptr<link> handle_;
	links_container links_;
	std::vector<std::string> allowed_otypes_;
	// temp guard until caf-based tree implementation is ready
	std::mutex links_guard_;
	using links_locker_t = std::lock_guard<std::mutex>;
};

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

