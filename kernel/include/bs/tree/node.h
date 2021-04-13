/// @author uentity
/// @date 15.09.2016
/// @brief BlueSky tree node class declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "bare_node.h"
#include "link.h"
#include "errors.h"
#include "../meta/is_container.h"
#include "../detail/enumops.h"

#include <optional>

NAMESPACE_BEGIN(blue_sky::tree)

class BS_API node : public engine {
public:
	using bare_type = bare_node;
	using engine_impl = node_impl;
	using engine_actor = node_actor;
	using weak_ptr = engine::weak_ptr<node>;
	using existing_index = bare_node::existing_index;
	using insert_status = bare_node::insert_status;

	/// Interface of node actor, you can only send messages matching it
	using actor_type = caf::typed_actor<
		// get home group
		caf::replies_to<a_home>::with<caf::group>,
		// get node's group ID
		caf::replies_to<a_home_id>::with<std::string>,
		// get node's handle
		caf::replies_to<a_node_handle>::with<link>,
		// get number of leafs
		caf::replies_to<a_node_size>::with<std::size_t>,
		// clone node
		caf::replies_to<a_clone, bool /* deep */>::with<node>,

		// obtain node's content sorted by given order
		caf::replies_to<a_node_leafs, Key /* order */>::with<links_v>,

		// obtain leafs keys sorted by given index
		caf::replies_to<a_node_keys, Key /* order */>::with<lids_v>,
		// sorted string keys
		caf::replies_to<a_node_keys, Key /* meaning */, Key /* order */>::with<std::vector<std::string>>,
		// sorted indexes (offsets)
		caf::replies_to<a_node_ikeys, Key /* order */>::with<std::vector<std::size_t>>,

		// find link by ID
		caf::replies_to<a_node_find, lid_type>::with<link>,
		// find link at specified index
		caf::replies_to<a_node_find, std::size_t>::with<link>,
		// find link by string key & meaning
		caf::replies_to<a_node_find, std::string, Key>::with<link>,

		// return link index
		caf::replies_to<a_node_index, lid_type>::with<existing_index>,
		// return index of link with string key & meaning
		caf::replies_to<a_node_index, std::string, Key>::with<existing_index>,

		// equal_range
		caf::replies_to<a_node_equal_range, std::string, Key>::with<links_v>,

		// deep search by ID (returns collection of 1 elem on success)
		caf::replies_to<a_node_deep_search, lid_type /* key */>::with<links_v>,
		// if last arg == true, then collects matching links over node's subtree
		caf::replies_to<a_node_deep_search, std::string, Key, bool /* return_first */>::with<links_v>,

		// insert new link
		caf::replies_to<a_node_insert, link, InsertPolicy>::with<insert_status>,
		// insert into specified position
		caf::replies_to<a_node_insert, link, std::size_t, InsertPolicy>::with<insert_status>,
		// insert bunch of links
		caf::replies_to<a_node_insert, links_v, InsertPolicy>::with<std::size_t>,

		// erase link by ID with specified options
		caf::replies_to<a_node_erase, lid_type>::with<std::size_t>,
		// erase link at specified position
		caf::replies_to<a_node_erase, std::size_t>::with<std::size_t>,
		// erase link with given string key & meaning
		caf::replies_to<a_node_erase, std::string, Key>::with<std::size_t>,
		// erase bunch of links
		caf::replies_to<a_node_erase, lids_v>::with<std::size_t>,
		// clears node content
		caf::replies_to<a_node_clear>::with<std::size_t>,

		// erase link by ID with specified options
		caf::replies_to<a_lnk_rename, lid_type, std::string>::with<std::size_t>,
		// erase link at specified position
		caf::replies_to<a_lnk_rename, std::size_t, std::string>::with<std::size_t>,
		// rename link(s) with specified name
		caf::replies_to<a_lnk_rename, std::string, std::string>::with<std::size_t>,

		// apply custom order
		caf::replies_to<a_node_rearrange, std::vector<std::size_t>>::with<error::box>,
		caf::replies_to<a_node_rearrange, lids_v>::with<error::box>,

		// run transaction in message queue of this node
		caf::replies_to<a_apply, node_transaction>::with<tr_result::box>
	>;

	///////////////////////////////////////////////////////////////////////////////
	//  constructors & maintance
	//
	/// create empty node and optionally add leafs
	node(links_v = {});
	explicit node(const bare_node& rhs);

	/// copy ctor & assignment
	node(const node& src) = default;
	auto operator=(const node& rhs) -> node& = default;
	auto operator=(const bare_node& rhs) -> node&;

	/// convert to bare link with direct access to internals (bypassing actor)
	auto bare() const -> bare_node;

	/// swap support
	friend auto swap(node& lhs, node& rhs) noexcept -> void {
		static_cast<engine&>(lhs).swap(rhs);
	}

	/// obtain/make node nil
	static auto nil() -> node;
	auto reset() -> void;

	/// test if node is nil
	auto is_nil() const -> bool;
	operator bool() const { return !is_nil(); }

	/// obtain owning link to this node
	/// [NOTE] only one owning link to node is allowed (multiple hard links to node are prihibited)
	auto handle() const -> link;

	/// return node's typed actor handle
	using engine::actor;
	auto actor() const -> actor_type {
		return engine::actor(*this);
	}

	/// `deep` flag is propagated to leafs
	auto clone(bool deep = false) const -> node;

	///////////////////////////////////////////////////////////////////////////////
	//  public API
	//
	/// number of elements in this node
	auto size() const -> std::size_t;

	/// check if node is empty
	auto empty() const -> bool;

	/// clears node
	// return number of cleared elements
	auto clear() const -> std::size_t;
	auto clear(launch_async_t) const -> void;

	/// get snapshot of node's content sorted with given order
	auto leafs(Key order = Key::AnyOrder) const -> links_v;

	/// obtain vector of link ID keys, sorted with given order
	auto keys(Key ordering = Key::AnyOrder) const -> lids_v;
	/// obtain vector of link indexes (offsets from beginning)
	auto ikeys(Key ordering = Key::AnyOrder) const -> std::vector<std::size_t>;
	/// obtain vector of leafs keys of `key_meaning` index type sorted by `ordering` key
	auto skeys(Key key_meaning, Key ordering = Key::AnyOrder) const -> std::vector<std::string>;

	// search link by given key
	auto find(std::size_t idx) const -> link;
	auto find(lid_type id) const -> link;
	/// find link by given key with specified treatment
	auto find(std::string key, Key key_meaning) const -> link;

	/// search among all subtree elements
	auto deep_search(lid_type id) const -> link;
	/// deep search by given key with specified treatment
	auto deep_search(std::string key, Key key_meaning) const -> link;
	/// collect matching links over node's subtree
	auto deep_equal_range(std::string key, Key key_meaning) const -> links_v;

	/// get integer index of a link relative to beginning
	auto index(lid_type lid) const -> existing_index;
	auto index(std::string key, Key key_meaning) const -> existing_index;

	/// find all links with given name, OID or type
	auto equal_range(std::string key, Key key_meaning) const -> links_v;

	/// leafs insertion
	auto insert(link l, InsertPolicy pol = InsertPolicy::AllowDupNames) const -> insert_status;
	/// insert link at given index
	auto insert(link l, std::size_t idx, InsertPolicy pol = InsertPolicy::AllowDupNames) const
	-> insert_status;
	/// insert bunch of links
	auto insert(links_v ls, InsertPolicy pol = InsertPolicy::AllowDupNames) const -> std::size_t;
	/// create and insert hard link that points to object
	auto insert(std::string name, sp_obj obj, InsertPolicy pol = InsertPolicy::AllowDupNames) const
	-> insert_status;
	/// create `objnode` with given node and insert hard link that points to it
	auto insert(std::string name, node N, InsertPolicy pol = InsertPolicy::AllowDupNames) const
	-> insert_status;

	/// insert links from given container
	/// [NOTE] container elements will be moved from passed container!
	template<typename C, typename = std::enable_if_t<meta::is_container_v<C>>>
	auto insert(C&& links, InsertPolicy pol = InsertPolicy::AllowDupNames) const -> void {
		for(auto& L : links) {
			static_assert(
				std::is_base_of<link, std::decay_t<decltype(L)>>::value,
				"Links container should contain shared pointers to `tree::link` objects"
			);
			insert(std::move(L), pol);
		}
	}

	/// leafs removal
	/// return removed leafs count
	auto erase(std::size_t idx) const -> std::size_t;
	auto erase(lid_type link_id) const -> std::size_t;
	/// erase leaf adressed by string key with specified treatment
	auto erase(std::string key, Key key_meaning) const -> std::size_t;
	/// erase bunch of leafs with given IDs
	auto erase(lids_v r) const -> std::size_t;

	/// rename link at given position
	auto rename(std::size_t idx, std::string new_name) const -> bool;
	/// rename link with given ID
	auto rename(lid_type lid, std::string new_name) const -> bool;
	/// rename link(s) with specified name
	auto rename(std::string old_name, std::string new_name) const -> std::size_t;

	/// apply custom order
	auto rearrange(std::vector<lid_type> new_order) const -> error;
	auto rearrange(std::vector<std::size_t> new_order) const -> error;

	/// applies functor to node atomically (invoke in node's queue)
	auto apply(node_transaction tr) const -> error;
	auto apply(launch_async_t, node_transaction tr) const -> void;

	///////////////////////////////////////////////////////////////////////////////
	//  events handling
	//
	using event_handler = std::function< void(node /* root */, event) >;

	/// returns ID of suscriber that is required for unsubscribe
	auto subscribe(event_handler f, Event listen_to = Event::All) const -> std::uint64_t;
	/// unsubscribe handlers from self & whole subtree
	auto unsubscribe(deep_t) const -> void;
	using engine::unsubscribe;

private:
	friend atomizer;
	friend weak_ptr;
	friend link_impl;
	friend node_impl;
	friend node_actor;

	using engine::operator=;

	auto pimpl() const -> node_impl*;

	auto start_engine() -> bool;

	node(sp_engine_impl impl);

	node(engine&&);
};

NAMESPACE_END(blue_sky::tree)

NAMESPACE_BEGIN(std)

/// support for engines in hashed containers
template<> struct hash<::blue_sky::tree::node> {
	auto operator()(const ::blue_sky::tree::node& N) const noexcept { return N.hash(); }
};

NAMESPACE_END(std)
