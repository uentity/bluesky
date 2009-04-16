#ifndef _BS_ARRAY_H_
#define _BS_ARRAY_H_

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

#include "bs_common.h"

template< class T >
class vector_traits {
	typedef std::vector< T > container;
	typedef typename container::value_type value_type;
	//! type of key - index
	typedef typename container::size_type key_type;
	//! type of iterator
	typedef typename container::iterator iterator;
	//! type of const iterator
	typedef typename container::const_iterator const_iterator;
	//references
	typedef typename container::reference reference;
	typedef typename container::const_reference const_reference;
	typedef typename container::size_type size_type;
};


template< class T >
class str_map_traits{
	typedef std::map< std::string, T > container;
	typedef T value_type;
	//! type of key - index
	typedef typename std::string key_type;
	//! type of iterator
	typedef typename container::iterator iterator;
	//! type of const iterator
	typedef typename container::const_iterator const_iterator;
	//references
	typedef typename container::mapped_type& reference;
	typedef const typename container::mapped_type& const_reference;
	typedef typename container::size_type size_type;
};

/// @brief BlueSky array base class
///
/// template params:
///           T -- type of array elements
/// cont_traits -- specifies underlying container
template< class T, template< class > cont_traits >
class bs_array : public objbase, public typename cont_traits< T >::container
{
public:
	//! type of value
	//typedef T value_type;
	typedef typename cont_traits::value_type value_type;
	//! type of vector of values
	typedef typename cont_traits::container container;
	//! type of key
	typedef typename cont_traits::size_type key_type;
	//! type of iterator
	typedef typename cont_traits::iterator iterator;
	//! type of const iterator
	typedef typename cont_traits::const_iterator const_iterator;
	//references
	typedef typename cont_traits::reference reference;
	typedef typename cont_traits::const_reference const_reference;
	typedef ulong size_type;

	// container should implement this
	
	/*!
		\brief Remove item method.
		\param key - key object
	 */
	virtual void erase(const key_type& k) {
		container::erase(k);
	}

	/*!
		\brief Add item method.

		Insert after key-index object.
		\param key - key object
		\param value - value object
		\return such as std::vector
	*/
	virtual bool insert(const key_type& key, const value_type& value) {
		return container::insert(key, value);
	}

	virtual typename cont_traits::iterator begin() {
		return container::begin();
	}

	virtual typename cont_traits::const_iterator begin() const {
		return container::begin();
	}

	virtual typename cont_traits::iterator end() {
		return container::end();
	}

	virtual typename cont_traits::const_iterator end() const {
		return container::end();
	}
	//virtual bool insert(const value_type& value) {
	//	return container::push_back(value);
	//}
	
	/// @brief Obtain array size
	/// 
	/// @return number of elements contained in array
	virtual size_type size() const {
		return container::size();
	}

	/// @brief Dynamic array resize
	/// 
	/// @param new_size New size of array
	virtual void resize(const size_type new_size) {
		container::resize(new_size);
	}

	/// @brief Items access operator (r/w)
	/// 
	/// @param key item key
	/// 
	/// @return modifyable refernce to element
	virtual reference operator[](const size_type key) {
		return container::operator[](key);
	}

	/// @brief Items access operator (r)
	/// 
	/// @param key item key
	/// 
	/// @return const refernce to element
	virtual const_refernce operator[](const size_type key) const {
		return container::operator[](key);
	}

	//! empty destructor
	virtual ~bs_array() {};

	//creation and copy functions definitions
	BLUE_SKY_TYPE_STD_CREATE_T_MEM(bs_array)
	BLUE_SKY_TYPE_STD_COPY_T_MEM(bs_array)

	BLUE_SKY_TYPE_DECL_T_MEM(bs_array, objbase, "bs_array",
		"Array of values of the same type indexed by integral type", "")
};

#endif

