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
struct vector_traits : public bs_vecbase_impl< T, std::vector< T > > {};

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
class bs_array : public cont_traits< T >::bs_array_base, public objbase {
public:
	// traits
	using cont_traits_t = cont_traits< T >;
	using base_t    = typename cont_traits_t::bs_array_base;
	using arrbase   = typename cont_traits_t::arrbase;
	using container = typename cont_traits_t::container;

	using sp_arrbase = typename arrbase::sp_arrbase;
	using sp_array   = std::shared_ptr< bs_array >;

	// inherited from bs_arrbase class
	using value_type = typename arrbase::value_type;
	using key_type   = typename arrbase::key_type;
	using size_type  = typename arrbase::size_type;

	using pointer                = typename arrbase::pointer;
	using reference              = typename arrbase::reference;
	using const_pointer          = typename arrbase::const_pointer;
	using const_reference        = typename arrbase::const_reference;
	using iterator               = typename arrbase::iterator;
	using const_iterator         = typename arrbase::const_iterator;
	using reverse_iterator       = typename arrbase::reverse_iterator;
	using const_reverse_iterator = typename arrbase::const_reverse_iterator;

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
		return std::make_shared< bs_array >(
			*std::static_pointer_cast< base_t >(base_t::clone())
		);
	}

	void swap(bs_array& arr) {
		objbase::swap(arr);
		base_t::swap(arr);
	}

	// if we assign arrays of same type - forward to trait's specific assignment operator
	bs_array& operator=(const bs_array& rhs) {
		assign_impl(rhs, std::true_type());
		return *this;
	}

	template< class R, template< class > class r_traits >
	bs_array& operator=(const bs_array< R, r_traits >& rhs) {
		assign(rhs);
		return *this;
	}

	/// @brief assign from array of castable type
	///
	/// @tparam R type of rhs array
	/// @param rhs source of assignment
	template< class R, template< class > class r_traits >
	void assign(const bs_array< R, r_traits >& rhs) {
		// dispatch if rhs trats is derived from cont_traits_t
		assign_impl(
			rhs,
			typename std::is_base_of<std::decay<cont_traits_t>, std::decay_t<r_traits<R>>>::type()
		);
	}

protected:
	// if we assign arrays of same type and castable traits - forward to trait's specific assignment operator
	template< class R, template< class > class r_traits >
	void assign_impl(const bs_array< R, r_traits >& rhs, std::true_type) {
		base_t::operator=(rhs);
	}

	/// @brief assign from array of castable type
	///
	/// @tparam R type of rhs array
	/// @param rhs source of assignment
	template< class R, template< class > class r_traits >
	void assign_impl(const bs_array< R, r_traits >& rhs, std::false_type) {
		// sanity
		if((void*)this->begin() == (void*)rhs.begin()) return;

		if(this->size() != rhs.size()) {
			// assign through swap
			bs_array(rhs.begin(), rhs.end()).swap(*this);
		}
		else {
			std::copy(rhs.begin(), rhs.end(), this->begin());
		}
	}

	BS_TYPE_DECL_INL_BEGIN(bs_array, objbase, "", \
			"Array of values of the same type indexed by integral type")
		td.add_constructor< bs_array, size_type >();
		td.add_constructor< bs_array, size_type, const value_type& >();
		// add the same ctors but with size specified as `int`
		// useful when size is specified as literal
		td.add_constructor< bs_array, int >();
		td.add_constructor< bs_array, int, const value_type& >();
		td.add_constructor< bs_array, const_iterator, const_iterator >();
	BS_TYPE_DECL_INL_END
};

}	// end of blue_sky namespace

