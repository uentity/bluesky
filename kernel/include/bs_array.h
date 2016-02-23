/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _BS_ARRAY_H_
#define _BS_ARRAY_H_

#include "bs_vecbase.h"
#include "bs_array_shared.h"
#include "bs_vector_shared.h"
#include "bs_kernel.h"
//#include "shared_vector.h"
#include <vector>

#if !defined(BS_ARRAY_DEFAULT_TRAITS)
#define BS_ARRAY_DEFAULT_TRAITS bs_array_shared
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
class BS_API bs_array : public objbase, public cont_traits< T >::bs_array_base
{
public:
	// traits
	typedef cont_traits< T > cont_traits_t;
	typedef typename cont_traits_t::arrbase arrbase_t;
	typedef typename cont_traits_t::container container;
	typedef typename cont_traits_t::bs_array_base base_t;

	typedef typename arrbase_t::sp_arrbase sp_arrbase;
	typedef bs_array< T, cont_traits > this_t;
	typedef smart_ptr< this_t, true > sp_array;

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

	// copy construct from container
	template< class R >
	bs_array(const R& rhs)
		: base_t(rhs)
	{}

	// construct from given size & fill value
	bs_array(size_type sz, const value_type& v = value_type())
		: base_t(sz, v)
	{}

	void dispose() const {
		objbase::dispose();
	}

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
		sp_array clon_ = BS_KERNEL.create_object(bs_resolve_type());
		clon_->resize(this->size());
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
	//creation and copy functions definitions
	BLUE_SKY_TYPE_STD_CREATE_T_MEM(bs_array);
	BLUE_SKY_TYPE_STD_COPY_T_MEM(bs_array);

	// don't include std ctors -- we need custom copy ctor, so use IFACE version
	BLUE_SKY_TYPE_DECL_T_MEM_IFACE(bs_array, objbase, "bs_array",
		"Array of values of the same type indexed by integral type", "");

	// default bs_array ctor implementation
	bs_array(bs_type_ctor_param /* param */)
		: bs_refcounter(), base_t()
	{}
	// default bs_array copy ctor implementation
	bs_array(const bs_array& src)
		: bs_refcounter(), objbase(src), base_t(src)
	{}
};

}	// end of blue_sky namespace

#endif

