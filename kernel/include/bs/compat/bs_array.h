/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "bs_vecbase.h"
//#include "bs_array_shared.h"
//#include "bs_vector_shared.h"
#include "../kernel.h"
#include "../type_macro.h"
//#include "shared_vector.h"
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
class BS_API bs_array : public cont_traits< T >::bs_array_base, public objbase
{
public:
	// traits
	typedef cont_traits< T > cont_traits_t;
	typedef typename cont_traits_t::arrbase arrbase_t;
	typedef typename cont_traits_t::container container;
	typedef typename cont_traits_t::bs_array_base base_t;

	typedef typename arrbase_t::sp_arrbase sp_arrbase;
	typedef bs_array< T, cont_traits > this_t;
	typedef std::shared_ptr< this_t > sp_array;

	// inherited from bs_arrbase class
	typedef typename arrbase_t::value_type value_type;
	typedef typename arrbase_t::size_type  size_type;
	typedef typename arrbase_t::key_type   key_type;

	typedef typename arrbase_t::pointer                pointer;
	typedef typename arrbase_t::reference              reference;
	typedef typename arrbase_t::const_pointer          const_pointer;
	typedef typename arrbase_t::const_reference        const_reference;
	typedef typename arrbase_t::iterator               iterator;
	typedef typename arrbase_t::const_iterator         const_iterator;
	typedef typename arrbase_t::reverse_iterator       reverse_iterator;
	typedef typename arrbase_t::const_reverse_iterator const_reverse_iterator;

	using arrbase_t::assign;

	bs_array() = default;
	bs_array(const bs_array&) = default;
	bs_array(bs_array&&) = default;

	// perfect forwarding ctor
	template < typename... Args >
	bs_array(Args&&... args) : base_t(std::forward< Args >(args)...) {}

	// copy construct from container
	//template< class R >
	//bs_array(const R& rhs)
	//	: base_t(rhs)
	//{}

	//// construct from given size & fill value
	//bs_array(size_type sz, const value_type& v = value_type())
	//	: base_t(sz, v)
	//{}

	//bs_array(const_iterator from, const_iterator to)
	//	: base_t(std::move(from), std::move(to))
	//{}

	// ctor via init
	template< class R >
	void init(const R& rhs) {
		this_t(rhs).swap(*this);
	}

	void init(size_type sz, const value_type& v = value_type()) {
		this_t(sz, v).swap(*this);
	}

	template< class input_iterator >
	void init(input_iterator start, input_iterator finish) {
		this_t(container(start, finish)).swap(*this);
	}

	// make array with copied data
	sp_arrbase clone() const {
		sp_array clon_ = std::make_shared< bs_array >(this->size());
		std::copy(this->begin(), this->end(), clon_->begin());
		return clon_;
	}

	void swap(bs_array& arr) {
		objbase::swap(arr);
		base_t::swap(arr);
	}

	template< class R, template< class > class r_traits >
	bs_array& operator=(const bs_array< R, r_traits >& rhs) {
		this->assign(rhs);
		return *this;
	}

protected:
	BS_TYPE_DECL_INL_BEGIN(bs_array, objbase, "bs_array", \
			"Array of values of the same type indexed by integral type", true, true)
		td.add_constructor< bs_array, size_type, const value_type& >();
		td.add_constructor< bs_array, const_iterator, const_iterator >();
	BS_TYPE_DECL_INL_END
};

}	// end of blue_sky namespace

