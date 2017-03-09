/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "vecbase.h"
#include "vecbase_shared.h"
#include "../type_descriptor.h"
#include "../type_macro.h"
#include "../objbase.h"
#include <vector>

#if !defined(BS_ARRAY_DEFAULT_TRAITS)
#define BS_ARRAY_DEFAULT_TRAITS vector_traits
#endif

namespace blue_sky {

/// @brief traits for arrays with std::vector container
template< class T >
struct BS_API vector_traits : public bs_vecbase_impl< T, std::vector< T > > {};

/*-----------------------------------------------------------------------------
 *  bs_array - BlueSky class to choose arrbase_impl or vecbase_impl for different containers
 *-----------------------------------------------------------------------------*/
/// @brief Contains array of key-value pairs with integral key (index)
///
/// bs_array expects std::vector-like syntax of container in traits
/// template params:
///           T -- type of array elements
/// cont_traits -- specifies underlying container
template< class T, template< class > class cont_traits = BS_ARRAY_DEFAULT_TRAITS >
class BS_API bs_array : public cont_traits< T >::bs_array_base, public objbase {
public:
	// traits
	typedef cont_traits< T > cont_traits_t;
	using base_t    = typename cont_traits_t::bs_array_base;
	using arrbase   = typename cont_traits_t::arrbase;
	using container = typename cont_traits_t::container;

	using typename arrbase::sp_arrbase;
	using sp_array = std::shared_ptr< bs_array >;

	// inherited from bs_arrbase class
	using typename arrbase::value_type;
	using typename arrbase::key_type;
	using typename arrbase::size_type;

	using typename arrbase::pointer;
	using typename arrbase::reference;
	using typename arrbase::const_pointer;
	using typename arrbase::const_reference;
	using typename arrbase::iterator;
	using typename arrbase::const_iterator;
	using typename arrbase::reverse_iterator;
	using typename arrbase::const_reverse_iterator;

	using arrbase::assign;

	bs_array() = default;
	bs_array(const bs_array&) = default;
	//bs_array(bs_array&&) = default;

	// perfect forwarding ctor
	template < typename... Args >
	bs_array(Args&&... args)
		: base_t(std::forward< Args >(args)...), objbase()
	{}

	// perfect forwarding init via swap
	template< typename... Args >
	void init(Args&&... args) {
		bs_array(std::forward< Args >(args)...).swap(*this);
	}

	// make array with copied data
	sp_arrbase clone() const {
		return std::make_shared< bs_array >(*this);
	}

	void swap(bs_array& arr) {
		objbase::swap(arr);
		base_t::swap(arr);
	}

	bs_array& operator=(const bs_array& rhs) {
		this->assign(rhs);
		return *this;
	}

	template< class R, template< class > class r_traits >
	bs_array& operator=(const bs_array< R, r_traits >& rhs) {
		this->assign(rhs);
		return *this;
	}

protected:
	BS_TYPE_DECL_INL_BEGIN(bs_array, objbase, "bs_array", \
			"Array of values of the same type indexed by integral type", true, true)
		td.add_constructor< bs_array, size_type >();
		td.add_constructor< bs_array, size_type, const value_type& >();
		td.add_constructor< bs_array, const_iterator, const_iterator >();
	BS_TYPE_DECL_INL_END
};

}	// end of blue_sky namespace

