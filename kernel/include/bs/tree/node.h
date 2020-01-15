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

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/mem_fun.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

// global alias to shorten typing
namespace mi = boost::multi_index;

class node_impl;
class node_actor;

class BS_API node : public objbase {
public:
	using id_type = link::id_type;

	// links are sorted by unique ID
	using id_key = mi::const_mem_fun<
		link, id_type, &link::id
	>;
	// and non-unique name
	using name_key = mi::const_mem_fun<
		link, std::string, &link::name
	>;
	// and non-unique object ID
	using oid_key = mi::const_mem_fun<
		link, std::string, &link::oid
	>;
	// and non-unique object type
	using type_key = mi::const_mem_fun<
		link, std::string, &link::obj_type_id
	>;
	// and have random-access index that preserve custom items ordering
	struct any_order {};

	// container that will store all node elements (links)
	using links_container = mi::multi_index_container<
		sp_link,
		mi::indexed_by<
			mi::sequenced< mi::tag< any_order > >,
			mi::hashed_unique< mi::tag< id_key >, id_key >,
			mi::ordered_non_unique< mi::tag< name_key >, name_key >,
			mi::ordered_non_unique< mi::tag< oid_key >, oid_key >,
			mi::ordered_non_unique< mi::tag< type_key >, type_key >
		>
	>;

	// key alias
	enum class Key { ID, OID, Name, Type, AnyOrder };
	template<Key K> using Key_const = std::integral_constant<Key, K>;

private:
	// convert from key alias -> key type
	template<Key K, class _ = void>
	struct Key_dispatch {
		using tag = id_key;
		using type = id_type;
	};
	template<class _>
	struct Key_dispatch<Key::Name, _> {
		using tag = name_key;
		using type = std::string;
	};
	template<class _>
	struct Key_dispatch<Key::OID, _> {
		using tag = oid_key;
		using type = std::string;
	};
	template<class _>
	struct Key_dispatch<Key::Type, _> {
		using tag = type_key;
		using type = std::string;
	};
	template<class _>
	struct Key_dispatch<Key::AnyOrder, _> {
		using tag = any_order;
		using type = std::size_t;
	};

public:
	template<Key K> using Key_tag = typename Key_dispatch<K>::tag;
	template<Key K> using Key_type = typename Key_dispatch<K>::type;

	// some useful type aliases
	template<Key K = Key::AnyOrder> using Index = typename links_container::index<Key_tag<K>>::type;
	template<Key K = Key::AnyOrder> using iterator = typename Index<K>::iterator;
	template<Key K = Key::AnyOrder> using const_iterator = typename Index<K>::const_iterator;
	template<Key K = Key::ID> using insert_status = std::pair<iterator<K>, bool>;

	/// range is a pair that supports iteration
	template<typename Iterator>
	struct range_t : public std::pair<Iterator, Iterator> {
		using base_t = std::pair<Iterator, Iterator>;
		using base_t::base_t;
		range_t(const range_t&) = default;
		range_t(range_t&&) = default;
		range_t(const base_t& rhs) : base_t(rhs) {}

		auto begin() const { return this->first; }
		auto end() const { return this->second; }

		auto export_lids() const -> std::vector<link::id_type> {
			auto sz = std::distance(begin(), end());
			if(sz <= 0) return {};
			auto res = std::vector<link::id_type>((size_t)sz);
			std::transform(
				begin(), end(), res.begin(),
				[](const auto& x) { return x->id(); }
			);
			return res;
		}
	};
	template<Key K = Key::ID> using range = range_t<iterator<K>>;
	template<Key K = Key::ID> using const_range = range_t<const_iterator<K>>;

	/// links insertions policy
	enum class InsertPolicy {
		AllowDupNames = 0,
		DenyDupNames = 1,
		RenameDup = 2,
		DenyDupOID = 4,
		ReplaceDupOID = 8,
		Merge = 16
	};
	/// link erase options (used for acctor)
	enum class EraseOpts { Normal = 0, Silent = 1, DontResetOwner = 2 };

	using actor_insert_status = std::pair<std::optional<std::size_t>, bool>;

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
		caf::replies_to<a_lnk_find, id_type>::with<sp_link>,

		// insert new link
		caf::replies_to<a_lnk_insert, sp_link, InsertPolicy>::with<actor_insert_status>,
		// insert into specified position
		caf::replies_to<a_lnk_insert, sp_link, std::size_t, InsertPolicy>::with<actor_insert_status>,
		// insert bunch of links
		caf::replies_to<a_lnk_insert, links_v, InsertPolicy>::with<std::size_t>,

		// erase link by ID with specified options
		caf::replies_to<a_lnk_erase, id_type, EraseOpts>::with<std::size_t>,
		// erase link at specified position
		caf::replies_to<a_lnk_erase, std::size_t>::with<std::size_t>,
		// erase link with given string name
		caf::replies_to<a_lnk_erase, std::string, Key>::with<std::size_t>,
		// erase bunch of links
		caf::replies_to<a_lnk_erase, std::vector<id_type>>::with<std::size_t>,

		// apply custom order
		caf::reacts_to<a_node_rearrange, std::vector<std::size_t>>,
		caf::reacts_to<a_node_rearrange, std::vector<id_type>>
	>;

public:
	/// Main API

	// return node's actor handle
	auto actor() const {
		return caf::actor_cast<actor_type>(actor_);
	}
	static auto actor(const node& N) {
		return N.actor();
	}

	/// number of elements in this node
	std::size_t size() const;

	/// check if node is empty
	bool empty() const;

	/// clears node
	void clear();

	/// get snapshot of node's content
	auto leafs(Key order = Key::AnyOrder) const -> links_v;

	// iterate in IDs order
	template<Key K = Key::AnyOrder>
	iterator<K> begin() const {
		return begin(Key_const<K>());
	}

	template<Key K = Key::AnyOrder>
	iterator<K> end() const {
		return end(Key_const<K>());
	}
	// non-template versions to support STL iteration features
	iterator<> begin() const {
		return begin<>();
	}
	iterator<> end() const {
		return end<>();
	}

	// search link by given key
	iterator<Key::AnyOrder> find(const std::size_t idx) const;
	iterator<Key::AnyOrder> find(const id_type& id) const;
	/// find link by given key with specified treatment
	iterator<Key::AnyOrder> find(const std::string& key, Key key_meaning = Key::ID) const;

	/// returns link instead of iterator
	template<Key K>
	sp_link search(const Key_type<K>& k) const {
		auto i = find(k);
		if(i == end<K>()) throw error("node::search", Error::KeyMismatch);
		return *i;
	}

	/// get integer index of a link relative to beginning
	std::size_t index(const id_type& lid) const;
	std::size_t index(const iterator<Key::AnyOrder>& lid) const;
	std::size_t index(const std::string& key, Key key_meaning) const;

	/// search among all subtree elements
	sp_link deep_search(const id_type& id) const;
	/// deep search by given key with specified treatment
	sp_link deep_search(const std::string& key, Key key_meaning) const;

	range<Key::Name> equal_range(const std::string& link_name) const;
	range<Key::OID>  equal_range_oid(const std::string& oid) const;
	range<Key::Type> equal_type(const std::string& type_id) const;

	/// leafs insertion
	insert_status<Key::AnyOrder> insert(sp_link l, InsertPolicy pol = InsertPolicy::AllowDupNames);
	/// insert link just before given position
	insert_status<Key::AnyOrder> insert(sp_link l, iterator<> pos, InsertPolicy pol = InsertPolicy::AllowDupNames);
	/// insert link at given index
	insert_status<Key::AnyOrder> insert(sp_link l, std::size_t idx, InsertPolicy pol = InsertPolicy::AllowDupNames);
	/// auto-create and insert hard link that points to object
	insert_status<Key::AnyOrder> insert(std::string name, sp_obj obj, InsertPolicy pol = InsertPolicy::AllowDupNames);
	/// insert links from given container
	/// [NOTE] container elements will be moved from passed container!
	template<
		typename C,
		typename = std::enable_if_t<meta::is_container_v<C>>
	>
	void insert(C&& links, InsertPolicy pol = InsertPolicy::AllowDupNames) {
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
	size_t erase(const std::size_t idx);
	size_t erase(const id_type& link_id);
	/// erase leaf adressed by string key with specified treatment
	size_t erase(const std::string& key, Key key_meaning);

	size_t erase(const range<Key::AnyOrder>& r);
	size_t erase(const range<Key::ID>& r);
	size_t erase(const range<Key::Name>& r);
	size_t erase(const range<Key::OID>& r);

	template<Key K>
	void erase(iterator<K> pos) {
		erase(range<K>{pos, std::advance(pos)});
	}

	/// obtain vector of keys for given index type
	template<Key K = Key::ID>
	std::vector<Key_type<K>> keys() const {
		static_assert(K != Key::AnyOrder, "There are no keys for custom order index");
		return keys(Key_const<K>());
	}

	/// rename link at given position
	bool rename(iterator<Key::AnyOrder> pos, std::string new_name);
	bool rename(const std::size_t idx, std::string new_name);
	/// rename link with given ID
	bool rename(const id_type& lid, std::string new_name);
	/// rename link adresses by given key
	/// if `all == true`, rename all links matced by key, otherwise first found link is renamed
	std::size_t rename(
		const std::string& key, std::string new_name, Key key_meaning = Key::ID, bool all = false
	);

	/// project any given iterator into custom order
	iterator<Key::AnyOrder> project(iterator<Key::ID>) const;
	iterator<Key::AnyOrder> project(iterator<Key::Name>) const;
	iterator<Key::AnyOrder> project(iterator<Key::OID>) const;
	iterator<Key::AnyOrder> project(iterator<Key::Type>) const;

	/// apply custom order
	auto rearrange(std::vector<id_type> new_order) -> void;
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

	/// API for managing link filters
	/// test is given link can be inserted into the node
	bool accepts(const sp_link& what) const;
	/// only simple object type filtera are supported now
	void accept_object_types(std::vector<std::string> allowed_types);
	std::vector<std::string> allowed_object_types() const;

	/// obtain link to this node conained in owner (parent) node
	/// [NOTE] only one owner node is allowed (multiple hard links to node are prihibited)
	sp_link handle() const;

	/// ensure that owner of all contained leafs is correctly set to this node
	/// if deep is true, correct owners in all subtree
	void propagate_owner(bool deep = false);

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
	// strong ref to node's actor
	caf::actor actor_;

	// set node's handle
	void set_handle(const sp_link& handle);

	/// accept link impl and optionally start internal actor
	node(bool start_actor, std::string custom_oid = "");
	/// maually start internal actor (if not started already)
	auto start_engine(const std::string& gid = "") -> bool;

	/// Implementation details
	iterator<Key::ID> begin(Key_const<Key::ID>) const;
	iterator<Key::Name> begin(Key_const<Key::Name>) const;
	iterator<Key::OID> begin(Key_const<Key::OID>) const;
	iterator<Key::Type> begin(Key_const<Key::Type>) const;
	iterator<Key::AnyOrder> begin(Key_const<Key::AnyOrder>) const;

	iterator<Key::ID> end(Key_const<Key::ID>) const;
	iterator<Key::Name> end(Key_const<Key::Name>) const;
	iterator<Key::OID> end(Key_const<Key::OID>) const;
	iterator<Key::Type> end(Key_const<Key::Type>) const;
	iterator<Key::AnyOrder> end(Key_const<Key::AnyOrder>) const;

	std::vector<Key_type<Key::ID>> keys(Key_const<Key::ID>) const;
	std::vector<Key_type<Key::Name>> keys(Key_const<Key::Name>) const;
	std::vector<Key_type<Key::OID>> keys(Key_const<Key::OID>) const;
	std::vector<Key_type<Key::Type>> keys(Key_const<Key::Type>) const;

	BS_TYPE_DECL
};
// handy aliases
using sp_node = std::shared_ptr<node>;
using sp_cnode = std::shared_ptr<const node>;
using node_ptr = object_ptr<node>;
using cnode_ptr = object_ptr<const node>;

NAMESPACE_END(blue_sky::tree)

// allow bitwise operations for InsertPoiicy enum class
BS_ALLOW_ENUMOPS(blue_sky::tree::node::InsertPolicy)
BS_ALLOW_ENUMOPS(blue_sky::tree::node::EraseOpts)

