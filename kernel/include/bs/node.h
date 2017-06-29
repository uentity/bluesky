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

class BS_API bs_node : public objbase {
public:
	using id_type = bs_link::id_type;

	// links are sorted by unique ID
	using id_key = mi::member<
		bs_link, const id_type, &bs_link::id_
	>;
	// and non-unique name
	using name_key = mi::member<
		bs_link, const std::string, &bs_link::name_
	>;
	// container that will store all node elements (links)
	using links_container = mi::multi_index_container<
		sp_link,
		mi::indexed_by<
			mi::ordered_unique< mi::tag< id_key >, id_key >,
			mi::ordered_non_unique< mi::tag< name_key >, name_key >
		>
	>;

	using iterator = links_container::iterator;
	using name_iterator = links_container::index< name_key >::type::iterator;
	using name_range = std::pair< name_iterator, name_iterator >;

	using insert_ret_t = std::pair< iterator, bool >;

	/// number of elements in this node
	std::size_t size() const;

	/// check if node is empty
	bool empty() const;

	/// clears node
	void clear();

	// iterate in IDs order
	iterator begin() const;
	iterator end() const;

	// iterate in alphabetical order
	name_iterator begin_name() const;
	name_iterator end_name() const;

	iterator find(const id_type& id) const;
	iterator find(const std::string& name) const;
	/// caution: slow!
	iterator find(const sp_obj& obj) const;

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
	insert_ret_t insert(const sp_link& l);
	// auto-create and insert hard link that points to object
	insert_ret_t insert(const std::string& name, const sp_obj& obj);

	void erase(const std::string& name);
	void erase(const sp_link& l);
	void erase(const sp_obj& obj);

	void erase(iterator pos);
	void erase(iterator from, iterator to);

	/// rename given link

	/// ctor
	bs_node();

private:
	// PIMPL
	class node_impl;
	std::unique_ptr< node_impl > pimpl_;
};

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

