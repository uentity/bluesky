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
#include "bs_array_shared.h"
#include "shared_vector.h"
#include <vector>
#if defined(BSPY_EXPORTING) || defined(BSPY_EXPORTING_PLUGIN)
#include <pyublas/numpy.hpp>
//#include "python/bs_nparray.h"
#endif

namespace blue_sky {

namespace bs_private {

template< class T, class array_t >
struct BS_API arrbase_traits_impl : public bs_arrbase< T >, public array_t {
	typedef array_t container;
	typedef bs_arrbase< T > arrbase;
	typedef arrbase_traits_impl< T, array_t > bs_array_base;

	// default ctor
	arrbase_traits_impl() {}

	// ctor for constructing from container copy
	arrbase_traits_impl(const container& c) : array_t(c) {}
};

template< class T, class vector_t >
struct BS_API vecbase_traits_impl : public bs_vecbase< T >, public vector_t {
	typedef vector_t container;
	typedef bs_vecbase< T > arrbase;
	typedef vecbase_traits_impl< T, vector_t > bs_array_base;

	// inherited from bs_arrbase class
	typedef typename arrbase::value_type value_type;
	typedef typename arrbase::key_type key_type;

	// default ctor
	vecbase_traits_impl() {}

	// ctor from vector copy
	vecbase_traits_impl(const container& c) : vector_t(c) {}

	using container::size;

	bool insert(const key_type& key, const value_type& value) {
		if(key > size()) return false;
		container::insert(container::begin() + key, value);
		return true;
	}

	bool insert(const value_type& value) {
		container::push_back(value);
		return true;
	}

	void erase(const key_type& key)	{
		container::erase(container::begin() + key);
	}
};

}

/// @brief traits for arrays with std::vector container
template< class T >
struct BS_API vector_traits : public bs_private::vecbase_traits_impl< T, std::vector< T > > {};

/// @brief traits for arrays with shared_array container
template< class T >
struct BS_API shared_array_traits : public bs_private::arrbase_traits_impl< T, bs_array_shared< T > > {};

/// @brief traits for arrays with shared_vector container
template< class T >
struct BS_API shared_vector_traits : public bs_private::vecbase_traits_impl< T, shared_vector< T > > {};

#if defined(BSPY_EXPORTING) || defined(BSPY_EXPORTING_PLUGIN)
/// @brief traits for arrays with pyublas::numpy_array container
template< class T >
struct BS_API numpy_array_traits : public bs_private::arrbase_traits_impl< T, pyublas::numpy_array< T > >
{};
#endif

/*-----------------------------------------------------------------------------
 *  bs_array class
 *-----------------------------------------------------------------------------*/
/// @brief Contains array of key-value pairs with integral key (index)
///
/// bs_array expects std::vector-like syntax of container in traits
/// template params:
///           T -- type of array elements
/// cont_traits -- specifies underlying container
template<
	class T,
	template< class > class cont_traits = 
#if defined(BSPY_EXPORTING) || defined(BSPY_EXPORTING_PLUGIN)
	numpy_array_traits
#else
	shared_array_traits
#endif
	>
class BS_API bs_array : public objbase, public cont_traits< T >::bs_array_base
{
public:
	typedef cont_traits< T > cont_traits_t;
	typedef typename cont_traits_t::arrbase arrbase_t;
	typedef typename cont_traits_t::container container;
	typedef typename cont_traits_t::bs_array_base base_t;
	typedef bs_array< T, cont_traits > this_t;

	// inherited from bs_arrbase class
	typedef typename arrbase_t::value_type value_type;
	typedef typename arrbase_t::reference reference;
	typedef typename arrbase_t::const_reference const_reference;
	typedef typename arrbase_t::size_type size_type;
	typedef typename arrbase_t::key_type key_type;

	// inherited from cont_traits
	//typedef typename cont_traits_t::iterator iterator;
	//typedef typename cont_traits_t::const_iterator const_iterator;
	//typedef typename cont_traits_t::reverse_iterator reverse_iterator;
	//typedef typename cont_traits_t::const_reverse_iterator const_reverse_iterator;

	//using container::insert;

	// copy construct from container
	bs_array(const container& c)
		: base_t(c)
	{}

	// ctor via init
	void init(const container& c) {
		this_t(c).swap(*this);
	}

	// init methods are indirect constructors
	void init(size_type sz, const value_type& v = value_type()) {
		this_t(container(sz, v)).swap(*this);
	}

	/// @brief Obtain array size
	/// Overloads bs_arrbase method
	/// @return number of elements contained in array
	size_type size() const {
		return static_cast< size_type >(container::size());
	}

	/// @brief Items access operator (r/w)
	/// Overloads bs_arrbase method
	/// @param key --- item key
	///
	/// @return modifyable reference to element
	reference operator[](const key_type& key) {
		return container::operator[](key);
	}

	/// @brief Items access operator (r)
	/// Overloads bs_arrbase method
	/// @param key -- item key
	///
	/// @return const refernce to element
	const_reference operator[](const key_type& key) const {
		return container::operator[](key);
	}

	void resize(size_type new_size) {
		container::resize(new_size);
	}

	void swap(bs_array& arr) {
		objbase::swap(arr);
		base_t::swap(arr);
	}

protected:
	//creation and copy functions definitions
	BLUE_SKY_TYPE_STD_CREATE_T_MEM(bs_array);
	BLUE_SKY_TYPE_STD_COPY_T_MEM(bs_array);

	BLUE_SKY_TYPE_DECL_T_MEM(bs_array, objbase, "bs_array",
		"Array of values of the same type indexed by integral type", "");
};

// default bs_array ctor implementation
template< class T, template< class > class cont_traits >
bs_array< T, cont_traits >::bs_array(bs_type_ctor_param param)
: bs_refcounter() //, base_t(param)
{}

// default bs_array copy ctor implementation
template< class T, template< class > class cont_traits >
bs_array< T, cont_traits >::bs_array(const bs_array< T, cont_traits >& src)
: bs_refcounter(), objbase(src), base_t(src)
{}

}	// end of blue_sky namespace

#endif

