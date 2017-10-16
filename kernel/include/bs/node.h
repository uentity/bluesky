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

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/member.hpp>

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
	// key alias
	enum class Key { ID, Name };
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

public:
	template<Key K> using Key_tag = typename Key_dispatch<K>::tag;
	template<Key K> using Key_type = typename Key_dispatch<K>::type;


	// container that will store all node elements (links)
	using links_container = mi::multi_index_container<
		sp_link,
		mi::indexed_by<
			mi::ordered_unique< mi::tag< id_key >, id_key >,
			mi::ordered_non_unique< mi::tag< name_key >, name_key >
		>
	>;

	// some useful type aliases
	template<Key K = Key::ID> using iterator = typename links_container::index<Key_tag<K>>::type::iterator;
	template<Key K = Key::ID> using const_iterator = typename links_container::index<Key_tag<K>>::type::const_iterator;
	template<Key K = Key::ID> using insert_ret_t = std::pair<iterator<K>, bool>;
	using name_range = std::pair< iterator<Key::Name>, iterator<Key::Name> >;

private:
	/// Implementation details
	iterator<Key::ID> begin(Key_const<Key::ID>) const;
	iterator<Key::Name> begin(Key_const<Key::Name>) const;

	iterator<Key::ID> end(Key_const<Key::ID>) const;
	iterator<Key::Name> end(Key_const<Key::Name>) const;

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

	// search link by given key
	iterator<Key::ID> find(const id_type& id) const;
	iterator<Key::ID> find(const std::string& name) const;
	/// caution: slow!
	iterator<Key::ID> find(const sp_obj& obj) const;


	/// returns link pointer instead of iterator
	template< typename Key >
	sp_link search(const Key& k) const {
		return *find(k);
	}

	/// search among all subtree elements
	sp_link deep_search(const id_type& id) const;

	name_range equal_range(const std::string& name) const;
	name_range equal_range(const sp_link& l) const;

	/// insertion
	insert_ret_t<Key::ID> insert(const sp_link& l);
	// auto-create and insert hard link that points to object
	insert_ret_t<Key::ID> insert(const std::string& name, const sp_obj& obj);

	void erase(const std::string& name);
	void erase(const sp_link& l);
	void erase(const sp_obj& obj);

	void erase(iterator<Key::ID> pos);
	void erase(iterator<Key::ID> from, iterator<Key::ID> to);

	/// rename given link

	/// ctor
	node();

private:
	// PIMPL
	class node_impl;
	std::unique_ptr< node_impl > pimpl_;
};

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

