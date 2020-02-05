/// @file
/// @author uentity
/// @date 15.09.2016
/// @brief BlueSky tree node class declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../objbase.h"
#include "../tree/errors.h"
#include "../detail/is_container.h"
#include "../detail/enumops.h"
#include "link.h"

NAMESPACE_BEGIN(blue_sky::tree)

class node_impl;
class node_actor;

class BS_API node : public objbase {
public:
	// some useful type aliases
	using existing_index = std::optional<std::size_t>;
	using insert_status = std::pair<existing_index, bool>;

	/// Interface of node actor, you can only send messages matching it
	using actor_type = caf::typed_actor<
		// get node's group ID
		caf::replies_to<a_node_gid>::with<std::string>,
		// propagate owner on child links
		caf::reacts_to<a_node_propagate_owner, bool>,
		// get node's handle
		caf::replies_to<a_node_handle>::with<sp_link>,
		// get number of leafs
		caf::replies_to<a_node_size>::with<std::size_t>,

		// obtain node's content
		caf::replies_to<a_node_leafs, Key>::with<links_v>,

		// find link by ID
		caf::replies_to<a_node_find, lid_type>::with<sp_link>,
		// find link at specified index
		caf::replies_to<a_node_find, std::size_t>::with<sp_link>,
		// find link by string key & meaning
		caf::replies_to<a_node_find, std::string, Key>::with<sp_link>,

		// deep search
		caf::replies_to<a_node_deep_search, lid_type>::with<sp_link>,
		caf::replies_to<a_node_deep_search, std::string, Key>::with<sp_link>,

		// return link index
		caf::replies_to<a_node_index, lid_type>::with<existing_index>,
		// return index of link with string key & meaning
		caf::replies_to<a_node_index, std::string, Key>::with<existing_index>,

		// equal_range
		caf::replies_to<a_node_equal_range, std::string, Key>::with<links_v>,

		// insert new link
		caf::replies_to<a_node_insert, sp_link, InsertPolicy>::with<insert_status>,
		// insert into specified position
		caf::replies_to<a_node_insert, sp_link, std::size_t, InsertPolicy>::with<insert_status>,
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
		caf::reacts_to<a_node_clear>,

		// erase link by ID with specified options
		caf::replies_to<a_lnk_rename, lid_type, std::string>::with<std::size_t>,
		// erase link at specified position
		caf::replies_to<a_lnk_rename, std::size_t, std::string>::with<std::size_t>,
		// rename link(s) with specified name
		caf::replies_to<a_lnk_rename, std::string, std::string>::with<std::size_t>,

		// apply custom order
		caf::reacts_to<a_node_rearrange, std::vector<std::size_t>>,
		caf::reacts_to<a_node_rearrange, lids_v>
	>;

public:
	/// Main API

	// return node's actor handle
	auto actor() const {
		return caf::actor_cast<actor_type>(raw_actor());
	}
	static auto actor(const node& N) {
		return N.actor();
	}

	/// number of elements in this node
	auto size() const -> std::size_t;

	/// check if node is empty
	auto empty() const -> bool;

	/// clears node
	auto clear() -> void;

	/// get snapshot of node's content
	auto leafs(Key order = Key::AnyOrder) const -> links_v;

	/// obtain vector of link ID keys
	auto keys(Key ordering = Key::AnyOrder) const -> lids_v;
	/// obtain vector of string keys for given index type
	auto skeys(Key key_meaning, Key ordering = Key::AnyOrder) const -> std::vector<std::string>;

	// search link by given key
	auto find(std::size_t idx) const -> sp_link;
	auto find(lid_type id) const -> sp_link;
	/// find link by given key with specified treatment
	auto find(std::string key, Key key_meaning = Key::ID) const -> sp_link;

	/// search among all subtree elements
	auto deep_search(lid_type id) const -> sp_link;
	/// deep search by given key with specified treatment
	auto deep_search(std::string key, Key key_meaning) const -> sp_link;

	/// get integer index of a link relative to beginning
	auto index(lid_type lid) const -> existing_index;
	auto index(std::string key, Key key_meaning) const -> existing_index;

	/// find all links with given name, OID or type
	auto equal_range(std::string key, Key key_meaning) const -> links_v;

	/// leafs insertion
	auto insert(sp_link l, InsertPolicy pol = InsertPolicy::AllowDupNames) -> insert_status;
	/// insert link at given index
	auto insert(sp_link l, std::size_t idx, InsertPolicy pol = InsertPolicy::AllowDupNames) -> insert_status;
	/// auto-create and insert hard link that points to object
	auto insert(std::string name, sp_obj obj, InsertPolicy pol = InsertPolicy::AllowDupNames) -> insert_status;
	/// insert bunch of links
	auto insert(links_v ls, InsertPolicy pol = InsertPolicy::AllowDupNames) -> std::size_t;

	/// insert links from given container
	/// [NOTE] container elements will be moved from passed container!
	template<typename C, typename = std::enable_if_t<meta::is_container_v<C>>>
	auto insert(C&& links, InsertPolicy pol = InsertPolicy::AllowDupNames) -> void {
		for(auto& L : links) {
			static_assert(
				std::is_base_of<link, std::decay_t<decltype(*L)>>::value,
				"Links container should contain shared pointers to `tree::link` objects"
			);
			insert(std::move(L), pol);
		}
	}

	/// leafs removal
	/// return removed elemsnt count
	auto erase(std::size_t idx) -> std::size_t;
	auto erase(lid_type link_id) -> std::size_t;
	/// erase leaf adressed by string key with specified treatment
	auto erase(std::string key, Key key_meaning) -> std::size_t;
	/// erase bunch of leafs with given IDs
	auto erase(lids_v r) -> std::size_t;

	/// rename link at given position
	auto rename(std::size_t idx, std::string new_name) -> bool;
	/// rename link with given ID
	auto rename(lid_type lid, std::string new_name) -> bool;
	/// rename link(s) with specified name
	auto rename(std::string old_name, std::string new_name) -> std::size_t;

	/// apply custom order
	auto rearrange(std::vector<lid_type> new_order) -> void;
	auto rearrange(std::vector<std::size_t> new_order) -> void;

	///////////////////////////////////////////////////////////////////////////////
	//  track node events
	//
	using handle_event_cb = std::function< void(sp_node, Event, prop::propdict) >;

	/// returns ID of suscriber that is required for unsubscribe
	auto subscribe(handle_event_cb f, Event listen_to = Event::All) -> std::uint64_t;
	auto unsubscribe(std::uint64_t event_cb_id) -> void;

	// stops retranslating messages to this node
	// if `deep` is true, then also disconnect each subtree node
	auto disconnect(bool deep = true) -> void;

	/// obtain link to this node conained in owner (parent) node
	/// [NOTE] only one owner node is allowed (multiple hard links to node are prihibited)
	auto handle() const -> sp_link;

	/// ensure that owner of all contained leafs is correctly set to this node
	/// if deep is true, correct owners in all subtree
	auto propagate_owner(bool deep = false) -> void;

	/// obtain node's group ID
	auto gid() const -> std::string;

	/// ctor - creates hard self link with given name
	node(std::string custom_id = "");
	// copy ctor makes deep copy of contained links
	node(const node& src);

	virtual ~node();

private:
	friend class blue_sky::atomizer;
	friend class cereal::access;

	friend class link;
	friend class node_impl;
	friend class node_actor;

	// PIMPL
	std::shared_ptr<node_impl> pimpl_;
	// scoped actor for requests
	caf::scoped_actor factor_;

	/// accept link impl and optionally start internal actor
	node(bool start_actor, std::string custom_oid = "");

	/// return node's raw (dynamic-typed) actor handle
	auto raw_actor() const -> const caf::actor&;

	/// maually start internal actor (if not started already)
	auto start_engine(const std::string& gid = "") -> bool;

	// set node's handle
	auto set_handle(const sp_link& handle) -> void;

	BS_TYPE_DECL
};
// handy aliases
using sp_node = std::shared_ptr<node>;
using sp_cnode = std::shared_ptr<const node>;
using node_ptr = object_ptr<node>;
using cnode_ptr = object_ptr<const node>;

NAMESPACE_END(blue_sky::tree)
