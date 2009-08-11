// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

//
// C++ Interface: bs_shell
//
// Description:
//
//
// Author: Гагарин Александр Владимирович <GagarinAV@ufanipi.ru>, (C) 2008
//
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
