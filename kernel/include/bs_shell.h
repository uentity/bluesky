/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief BlueSky shell
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _BS_SHELL_H
#define _BS_SHELL_H

#include "bs_common.h"
#include "bs_tree.h"

#include <iterator>

#ifdef _MSC_VER
#pragma warning(push)
//disable complaints about std::iterator have no dll interface
#pragma warning(disable:4275)
#endif

namespace blue_sky {

//deep iterator can visit each item contained in given subtree
class BS_API deep_iterator : public std::iterator< std::bidirectional_iterator_tag, bs_link, ptrdiff_t,
	sp_link, sp_link::ref_t >
{
	friend class bs_shell;

public:
	typedef std::list< sp_link > path_t;
	//ctors
	//TODO: remove following ctor
	deep_iterator();
	//copy ctor
	deep_iterator(const deep_iterator& i);
	//iterator will point to current shells' directory
	deep_iterator(const smart_ptr< bs_shell, true >& shell);
	//construct from absolute path
	//deep_iterator(const std::string& abs_path);
	//construct from relative path to given shell
	//deep_iterator(const smart_ptr< bs_shell, true >& shell, const std::string& rel_path);

	reference operator*() const;

	pointer operator->() const;

	deep_iterator& operator++();
	deep_iterator operator++(int);

	deep_iterator& operator--();
	deep_iterator operator--(int);

	bool operator ==(const deep_iterator&) const;
	bool operator !=(const deep_iterator&) const;

	std::string full_name() const;
		//const path_t& path() const;
		//std::string path_name() const;

	bool jump_up();
	bool is_end() const;

private:
	class di_impl;
	st_smart_ptr< di_impl > pimpl_;
};

class BS_API bs_shell : public objbase {
	friend class deep_iterator;

public:
	typedef smart_ptr< bs_shell, true > sp_shell;

private:
	deep_iterator pos_;

	class shell_impl;
	smart_ptr< shell_impl, false > pimpl_;

	BLUE_SKY_TYPE_DECL(bs_shell)
};
//make typedef global
typedef bs_shell::sp_shell sp_shell;

}	//end of blue_sky namespace

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif	//guardian
