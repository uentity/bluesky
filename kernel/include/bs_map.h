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

#ifndef _BS_MAP_H_
#define _BS_MAP_H_

#include "bs_common.h"
#include "bs_object_base.h"
#include "bs_kernel.h"
#include <map>

//#include <map>
//#ifdef UNIX
//#include <ext/hash_map>
//#else
//#include <hash_map>
//#endif

 /*!
	 \brief Blue-sky map.
	 \param key_type = type of map's key (first)
	 \param value_type = type of map's values (second)
 */
#ifdef UNIX
	#define BS_MAP(key_type, value_type) std::map< key_type, value_type >
	//#define BS_MAP(key_type, value_type) __gnu_cxx::hash_map< key_type, value_type >
#else
	#define BS_MAP(key_type, value_type) std::map< key_type, value_type >
	//#define BS_MAP(key_type, value_type) stdext::hash_map< key_type, value_type >
#endif

namespace blue_sky {

/// @brief traits for maps with string key
template< class T >
struct str_val_traits : public BS_MAP(std::string, T) {
	/// container
	typedef BS_MAP(std::string, T) container;
	/// value type
	typedef typename container::mapped_type value_type;
	/// key type
	typedef std::string key_type;
	//! type of reference to second object (value)
	typedef typename container::mapped_type& reference;
	typedef const typename container::mapped_type& const_reference;

	//! type of map's iterator
	typedef typename container::iterator iterator;
	//! type of map's const iterator
	typedef typename container::const_iterator const_iterator;
};

/*-----------------------------------------------------------------------------
 *  bs_table class
 *-----------------------------------------------------------------------------*/
/// @brief Contains array of key-value pairs addressed by key
///
/// bs_map doesn't restrict container type and expect std::map-like container syntax
/// template params:
///           T -- type of array elements
/// cont_traits -- specifies underlying container
template< class T, template< class > class cont_traits >
class BS_API bs_map : public objbase, public cont_traits< T >::container
{
public:
	typedef cont_traits< T > cont_traits_t;

	//! type of vector of values
	typedef typename cont_traits_t::container container;
	typedef typename container::value_type data_pair;
	//! type of value
	typedef typename cont_traits_t::value_type value_type;
	//references
	typedef typename cont_traits_t::reference reference;
	typedef typename cont_traits_t::const_reference const_reference;
	//! type of key - index
	typedef typename cont_traits_t::size_type size_type;
	typedef typename cont_traits_t::key_type key_type;

	//! type of iterator
	typedef typename cont_traits_t::iterator iterator;
	//! type of const iterator
	typedef typename cont_traits_t::const_iterator const_iterator;

	using container::begin;
	using container::end;

	/// @brief Obtain array size
	/// Overloads bs_arrbase method
	/// @return number of elements contained in array
	size_type size() const {
		return static_cast< size_type >(container::size());
	}

	/// @brief Items access operator (r/w)
	/// @param key --- item key
	///
	/// @return modifyable refernce to element
	reference operator[](const key_type& key) {
		return container::operator[](key);
	}

	/// @brief Items access operator (r)
	/// @param key -- item key
	///
	/// @return const refernce to element
	const_reference operator[](const key_type& key) const {
		return at(key);
	}

	/*!
		\brief Search for item method.
		\param key - key object
	 */
	reference at(const key_type& key) {
		return _at< iterator, reference >(key);
	}

	const_reference at(const key_type& key) const {
		return const_cast< this_t* >(this)->_at< const_iterator, const_reference >(key);
	}
	

	/*!
	  \brief Add item method.

	  Insert after key-index object.
	  \param key - key object
	  \param value - value object
	  \return is item really added?
	  */
	bool insert(const key_type& key, const value_type& value) {
		return container::insert(data_pair(key, value)).second;
	}

	void erase(const key_type& key) {
		container::erase(key);
	}

private:
	typedef bs_map< T, cont_traits > this_t;

	template< class iterator_t, class ref_t >
	ref_t _at(const key_type& key) {
		iterator_t p_res(find(key));
		if(p_res != end())
			return p_res->second;
		else
			throw std::out_of_range(std::string("str_val_table: no element found with key = ") + key);
	}

	//creation and copy functions definitions
	BLUE_SKY_TYPE_STD_CREATE_T_MEM(bs_map);
	BLUE_SKY_TYPE_STD_COPY_T_MEM(bs_map);

	BLUE_SKY_TYPE_DECL_T_MEM(bs_map, objbase, "bs_map",
		"Map of values of the same type indexed by key", "");
};

// default bs_array ctor implementation
template< class T, template< class > class cont_traits >
bs_map< T, cont_traits >::bs_map(bs_type_ctor_param param)
: bs_refcounter(), objbase(param), container()
{}

// default bs_array copy ctor implementation
template< class T, template< class > class cont_traits >
bs_map< T, cont_traits >::bs_map(const bs_map< T, cont_traits >& src)
: bs_refcounter(), objbase(src), container(src)
{}

// make alias with 1 template param and str_val_traits accepted by default
template< class T >
class str_val_table : public bs_map< T, str_val_traits > {
	typedef bs_map< T, str_val_traits > base_t;
	
	//creation and copy functions definitions
	BLUE_SKY_TYPE_STD_CREATE_T_MEM(str_val_table);
	BLUE_SKY_TYPE_STD_COPY_T_MEM(str_val_table);

	BLUE_SKY_TYPE_DECL_T_MEM(str_val_table, objbase, "str_val_table",
		"Map of values of the same type indexed by string", "");
};

// standard ctor
template< class T >
str_val_table< T >::str_val_table(bs_type_ctor_param param)
	: bs_refcounter(), base_t(param)
{}

// copy ctor
template< class T >
str_val_table< T >::str_val_table(const str_val_table< T >& src)
	: bs_refcounter(), base_t(src)
{}

}	// end of blue_sky namespace

#endif

