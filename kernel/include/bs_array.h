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

#ifndef _BS_ARRAY_H_
#define _BS_ARRAY_H_

#include "bs_arrbase.h"
#include <map>


#ifdef UNIX
#define BS_MAP(key_type, value_type) std::map< key_type, value_type >
//#define BS_MAP(key_type, value_type) __gnu_cxx::hash_map< key_type, value_type >
#else
#define BS_MAP(key_type, value_type) std::map< key_type, value_type >
//#define BS_MAP(key_type, value_type) stdext::hash_map< key_type, value_type >
#endif

namespace blue_sky {
/*-----------------------------------------------------------------------------
 *  bs_array class
 *-----------------------------------------------------------------------------*/
/// @brief traits for arrays with std::vector container
template< class T >
struct vector_traits : public std::vector< T > {
	typedef std::vector< T > container;
	typedef typename container::value_type value_type;
	//! type of key - index
	typedef typename container::size_type size_type;
	typedef typename container::size_type key_type;
	//! type of iterator
	typedef typename container::iterator iterator;
	//! type of const iterator
	typedef typename container::const_iterator const_iterator;
	//references
	typedef typename container::reference reference;
	typedef typename container::const_reference const_reference;

	using container::size;
	using container::begin;
	using container::end;
	using container::erase;
	using container::push_back;
};

/// @brief Contains array of key-value pairs with integral key (index)
///
/// bs_array implies that container is std::vector-like
/// template params:
///           T -- type of array elements
/// cont_traits -- specifies underlying container
template< class T, template< class > class cont_traits >
class BS_API_PLUGIN bs_array_t : public bs_arrbase< T >, public cont_traits< T >
{
public:
	typedef bs_arrbase< T > arrbase_t;
	typedef cont_traits< T > cont_traits_t;

	// inherited from bs_arrbase class
	//! type of value
	typedef typename arrbase_t::value_type value_type;
	//references
	typedef typename arrbase_t::reference reference;
	typedef typename arrbase_t::const_reference const_reference;
	//! type of key - index
	typedef typename arrbase_t::size_type size_type;
	typedef typename arrbase_t::key_type key_type;
	//! type of iterator
	typedef typename arrbase_t::iterator iterator;
	typedef typename arrbase_t::template iterator_wrapper< value_type, typename cont_traits_t::iterator > iterator_w;
	//typedef typename cont_traits_t::iterator iterator;
	//! type of const iterator
	typedef typename arrbase_t::const_iterator const_iterator;
	typedef typename arrbase_t::template iterator_wrapper< const value_type, typename cont_traits_t::const_iterator > const_iterator_w;

	// inherited from cont_traits
	//! type of vector of values
	typedef typename cont_traits_t::container container;
	//typedef typename cont_traits_t::const_iterator const_iterator;

	using cont_traits_t::push_back;
	using cont_traits_t::begin;
	using cont_traits_t::end;

	/// @brief Obtain array size
	/// Overloads bs_arrbase method
	/// @return number of elements contained in array
	size_type size() const {
		return static_cast< size_type >(cont_traits_t::size());
	}

	/// @brief Items access operator (r/w)
	/// Overloads bs_arrbase method
	/// @param key --- item key
	/// 
	/// @return modifyable refernce to element
	reference operator[](const key_type& key) {
		return cont_traits_t::operator[](key);
	}

	/// @brief Items access operator (r)
	/// Overloads bs_arrbase method
	/// @param key -- item key
	/// 
	/// @return const refernce to element
	const_reference operator[](const key_type& key) const {
		return cont_traits_t::operator[](key);
	}

	/*!
	  \brief Add item method.

	  Insert after key-index object.
	  \param key - key object
	  \param value - value object
	  \return such as std::vector
	  */
	bool insert(const key_type& key, const value_type& value) {
		if(key > size()) return false;
		cont_traits_t::insert(begin() + key, value);
		return true;
	}

	/*!
	  \brief Add item method.

	  Push back insert method.
	  \param value - value object
	  \return such as std::map
	  */
	bool insert(const value_type& value)	{
		push_back(value);
		return true;
	}

	/*!
	  \brief Remove item method.
	  \param key - key object
	  */
	void erase(const key_type& key)	{
		cont_traits_t::erase(begin() + key);
	}

	// begin-end support for iteration through container
	iterator begin() {
		return iterator(new iterator_w(container::begin()));
	}

	const_iterator begin() const {
		return const_iterator(new const_iterator_w(container::begin()));
	}

	iterator end() {
		return iterator(new iterator_w(container::end()));
	}

	const_iterator end() const {
		return const_iterator(new const_iterator_w(container::end()));
	}

	//creation and copy functions definitions
	BLUE_SKY_TYPE_STD_CREATE_T_MEM(bs_array_t);
	BLUE_SKY_TYPE_STD_COPY_T_MEM(bs_array_t);

	BLUE_SKY_TYPE_DECL_T_MEM(bs_array_t, objbase, "bs_array_t",
		"Array of values of the same type indexed by integral type", "");
};

// default bs_array ctor implementation
template< class T, template< class > class cont_traits >
bs_array_t< T, cont_traits >::bs_array_t(bs_type_ctor_param param)
: bs_refcounter(), arrbase_t(param)
{}

// default bs_array copy ctor implementation
template< class T, template< class > class cont_traits >
bs_array_t< T, cont_traits >::bs_array_t(const bs_array_t< T, cont_traits >& src)
: bs_refcounter(), arrbase_t(src), cont_traits_t(src)
{}

}	// end of blue_sky namespace

#endif

