/// @file
/// @author uentity
/// @date 15.09.2016
/// @brief BlueSky tree node class declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "objbase.h"
#include "link.h"
#include "exception.h"

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/mem_fun.hpp>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)
// global alias to shorten typing
namespace mi = boost::multi_index;

class BS_API node : public objbase {
public:
	using id_type = link::id_type;

	// links are sorted by unique ID
	using id_key = mi::member<
		link, const id_type, &link::id_
	>;
	// and non-unique name
	using name_key = mi::member<
		link, const std::string, &link::name_
	>;
	// and non-unique object ID
	using oid_key = mi::const_mem_fun<
		link, std::string, &link::oid
	>;
	// key alias
	enum class Key { ID, OID, Name };
	template<Key K> using Key_const = std::integral_constant<Key, K>;

private:
	// convert from key alias -> key type
	template<Key K, class unused = void>
	struct Key_dispatch {
		using tag = id_key;
		using type = id_type;
	};
	template<class unused>
	struct Key_dispatch<Key::Name, unused> {
		using tag = name_key;
		using type = std::string;
	};
	template<class unused>
	struct Key_dispatch<Key::OID, unused> {
		using tag = oid_key;
		using type = std::string;
	};

public:
	template<Key K> using Key_tag = typename Key_dispatch<K>::tag;
	template<Key K> using Key_type = typename Key_dispatch<K>::type;

	// container that will store all node elements (links)
	using links_container = mi::multi_index_container<
		sp_link,
		mi::indexed_by<
			mi::ordered_unique< mi::tag< id_key >, id_key >,
			mi::ordered_non_unique< mi::tag< name_key >, name_key >,
			mi::ordered_non_unique< mi::tag< oid_key >, oid_key >
		>
	>;

	// some useful type aliases
	template<Key K = Key::ID> using iterator = typename links_container::index<Key_tag<K>>::type::iterator;
	template<Key K = Key::ID> using const_iterator = typename links_container::index<Key_tag<K>>::type::const_iterator;
	template<Key K = Key::ID> using insert_status = std::pair<iterator<K>, bool>;
	template<Key K = Key::ID> using range = std::pair<iterator<K>, iterator<K>>;
	template<Key K = Key::ID> using const_range = std::pair<const_iterator<K>, const_iterator<K>>;

public:
	/// Main API

	/// number of elements in this node
	std::size_t size() const;

	/// check if node is empty
	bool empty() const;

	/// clears node
	void clear();

	// iterate in IDs order
	template<Key K = Key::ID>
	iterator<K> begin() const {
		return begin(Key_const<K>());
	}

	template<Key K = Key::ID>
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
	iterator<Key::ID> find(const id_type& id) const;
	iterator<Key::ID> find(const std::string& link_name) const;
	/// caution: slow!
	// find link by given object ID (first found link is returned)
	iterator<Key::ID> find_oid(const std::string& oid) const;

	/// returns link pointer instead of iterator
	template<Key K>
	sp_link search(const Key_type<K>& k) const {
		auto i = find(k);
		if(i == end<K>()) throw bs_kexception("Unable to find link by given key", "node::search");
		return *i;
	}

	/// search among all subtree elements
	sp_link deep_search(const id_type& id) const;
	sp_link deep_search(const std::string& link_name) const;
	sp_link deep_search_oid(const std::string& oid) const;

	range<Key::Name> equal_range(const std::string& link_name) const;
	range<Key::OID> equal_range_oid(const std::string& oid) const;

	/// leafs insertion
	insert_status<Key::ID> insert(sp_link l);
	// auto-create and insert hard link that points to object
	insert_status<Key::ID> insert(std::string name, sp_obj obj);

	/// leafs removal
	void erase(const id_type& link_id);
	void erase(const std::string& link_name);
	void erase_oid(const std::string& oid);

	void erase(const range<Key::ID>& r);
	void erase(const range<Key::Name>& r);
	void erase(const range<Key::OID>& r);

	template<Key K>
	void erase(const iterator<K>& pos) {
		erase(range<K>{pos, std::advance(pos)});
	}

	/// rename given link

	/// ctor
	node();
	// copy ctor makes deep copy of contained links
	node(const node& src);

	virtual ~node();

private:
	// PIMPL
	class node_impl;
	std::unique_ptr< node_impl > pimpl_;

	/// Implementation details
	iterator<Key::ID> begin(Key_const<Key::ID>) const;
	iterator<Key::Name> begin(Key_const<Key::Name>) const;
	iterator<Key::OID> begin(Key_const<Key::OID>) const;

	iterator<Key::ID> end(Key_const<Key::ID>) const;
	iterator<Key::Name> end(Key_const<Key::Name>) const;
	iterator<Key::OID> end(Key_const<Key::OID>) const;

	BS_TYPE_DECL
};

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

