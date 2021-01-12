/// @date 22.09.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include "link.h"

NAMESPACE_BEGIN(blue_sky::tree)

class BS_API bare_node {
public:
	using existing_index = std::optional<std::size_t>;
	using insert_status = std::pair<existing_index, bool>;

	/// construct from `node` instance
	explicit bare_node(const node& rhs);
	auto operator=(const node& rhs) -> bare_node&;

	auto armed() const -> node;

	/// test if node is nil
	auto is_nil() const -> bool;
	operator bool() const { return !is_nil(); }

	/// obtain link to this node conained in owner (parent) node
	/// [NOTE] only one owner node is allowed (multiple hard links to node are prihibited)
	auto handle() const -> link;

	/// hash for appropriate containers
	auto hash() const noexcept -> std::size_t;

	/// swap spport
	friend auto swap(bare_node& lhs, bare_node& rhs) noexcept -> void {
		std::swap(lhs.pimpl_, rhs.pimpl_);
	}

	///////////////////////////////////////////////////////////////////////////////
	//  Main node API
	//
	/// number of elements in this node
	auto size() const -> std::size_t;
	
	/// check if node is empty
	auto empty() const -> bool;

	/// get snapshot of node's content sorted with given order
	auto leafs(Key order = Key::AnyOrder) const -> links_v;

	/// obtain vector of link ID keys, sorted with given order
	auto keys(Key ordering = Key::AnyOrder) const -> lids_v;
	/// obtain vector of link indexes (offsets from beginning)
	auto ikeys(Key ordering = Key::AnyOrder) const -> std::vector<std::size_t>;

	// search link by given key
	auto find(std::size_t idx) const -> link;
	auto find(lid_type id) const -> link;

	/// get integer index of a link relative to beginning
	auto index(lid_type lid) const -> existing_index;

	/// insert one link (safe for link)
	auto insert(link l, InsertPolicy pol = InsertPolicy::AllowDupNames) -> insert_status;
	/// can be used to insert just created new link not shared with anyone
	auto insert(unsafe_t, link l, InsertPolicy pol = InsertPolicy::AllowDupNames) -> insert_status;

	/// insert bunch of links (safe for links)
	auto insert(links_v ls, InsertPolicy pol = InsertPolicy::AllowDupNames) -> std::size_t;
	/// insert bunch of links (safe for links)
	auto insert(unsafe_t, links_v ls, InsertPolicy pol = InsertPolicy::AllowDupNames) -> std::size_t;

	/// create and insert hard link that points to object
	auto insert(std::string name, sp_obj obj, InsertPolicy pol = InsertPolicy::AllowDupNames)
	-> insert_status;
	/// create `objnode` with given node and insert hard link that points to it
	auto insert(std::string name, node N, InsertPolicy pol = InsertPolicy::AllowDupNames)
	-> insert_status;

	/// leafs removal
	/// return removed leafs count
	auto erase(std::size_t idx) -> std::size_t;
	auto erase(lid_type link_id) -> std::size_t;

	auto clear() -> std::size_t;

	/// apply custom order
	auto rearrange(std::vector<lid_type> new_order) -> error;
	auto rearrange(std::vector<std::size_t> new_order) -> error;

private:
	friend node;
	friend node_actor;

	std::shared_ptr<node_impl> pimpl_;

	explicit bare_node(std::shared_ptr<node_impl> impl);

	auto pimpl() const -> node_impl*;
};

NAMESPACE_END(blue_sky::tree)

NAMESPACE_BEGIN(std)

/// support for engines in hashed containers
template<> struct hash<::blue_sky::tree::bare_node> {
	auto operator()(const ::blue_sky::tree::bare_node& N) const noexcept { return N.hash(); }
};

NAMESPACE_END(std)
