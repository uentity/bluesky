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

#ifndef _BS_ARRBASE_H_
#define _BS_ARRBASE_H_

#include "bs_common.h"

namespace blue_sky {

/*-----------------------------------------------------------------------------
 *  bs_arrbase -- base class of BlueSky arrays
 *-----------------------------------------------------------------------------*/

/// @brief Base class of all BlueSky arrays with Python numpy support
/// Expected that underlying container is std::vector-like, i.e.
/// all elements are stored in continuos memory segment
template< class T >
class BS_API bs_arrbase : virtual public bs_refcounter {
public:
	typedef bs_arrbase< T > this_t;
	typedef smart_ptr < this_t, true > sp_arrbase;

	typedef T value_type;
	typedef ulong size_type;
	typedef size_type key_type;

	typedef value_type& reference;
	typedef const value_type& const_reference;

	/// @brief Obtain array size
	///
	/// @return number of elements contained in array
	virtual size_type size() const = 0;

	/// @brief Subscripting operator
	/// Forward call to ss(key)
	/// @param key item key
	/// 
	/// @return modifiable reference to element
	virtual reference operator[](const key_type& key) = 0;

	/// @brief Items access function (r/w) - syntax sugar for accessing via pointer
	///
	/// @param key item key
	///
	/// @return modifiable reference to element
	reference ss(const key_type& key) {
		return operator[](key);
	}

	/// @brief Subscripting operator
	/// Forward call to ss(key)
	/// @param key item key
	/// 
	/// @return modifiable reference to element
	virtual const_reference operator[](const key_type& key) const = 0;

	/// @brief Items access function (r) - syntax sugar for accessing via pointer
	///
	/// @param key item key
	///
	/// @return const reference to element
	const_reference ss(const key_type& key) const {
		return operator[](key);
	}

	virtual void resize(size_type new_size) = 0;

	//virtual sp_arrbase clone() const = 0;

	/// @brief empty destructor
	virtual ~bs_arrbase() {};

	//BS_COMMON_DECL_T_MEM(bs_arrbase);
	//BS_LOCK_THIS_DECL(bs_arrbase);

	//BLUE_SKY_TYPE_DECL_T_MEM(bs_arrbase, objbase, "bs_arrbase",
	//	"Base class for BlueSky continuous array of values of the same type indexed by integral type", "");
};

//template< class T >
//bs_arrbase< T >::bs_arrbase(bs_type_ctor_param param)
//	: bs_refcounter(), objbase(param)
//{}
//
//// copy ctor
//template< class T >
//bs_arrbase< T >::bs_arrbase(const bs_arrbase& a)
//	: bs_refcounter(), objbase(a)
//{}

/*-----------------------------------------------------------------------------
 *  bs_vecbase -- base class of BlueSky vectors
 *-----------------------------------------------------------------------------*/
// difference from arrbase is insert and erase operations
template< class T >
class BS_API bs_vecbase : public bs_arrbase< T > {
public:
	typedef bs_arrbase< T > arrbase_t;
	typedef typename arrbase_t::key_type key_type;
	typedef typename arrbase_t::value_type value_type;

	virtual bool insert(const key_type& key, const value_type& value) = 0;
	virtual bool insert(const value_type& value) = 0;
	virtual void erase(const key_type& key) = 0;

	/// @brief empty destructor
	virtual ~bs_vecbase() {};

	//using arrbase_t::bs_type;

	//BS_COMMON_DECL_T_MEM(bs_vecbase);
	//BS_LOCK_THIS_DECL(bs_vecbase);
};

//template< class T >
//bs_vecbase< T >::bs_vecbase(bs_type_ctor_param param)
//	: bs_refcounter(), arrbase_t(param)
//{}
//
//// copy ctor
//template< class T >
//bs_vecbase< T >::bs_vecbase(const bs_vecbase& a)
//	: bs_refcounter(), arrbase_t(a)
//{}

}	// namespace blue-sky
#endif	// file guard

