/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_NPVEC_SERIALIZE_DUQHKGZW
#define BS_NPVEC_SERIALIZE_DUQHKGZW

#include "bs_npvec.h"
#include "bs_npvec_shared.h"

#include "bs_serialize_macro.h"
#include "bs_serialize_overl.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/array.hpp>
#include <boost/archive/polymorphic_iarchive.hpp>
#include <boost/archive/polymorphic_oarchive.hpp>

/*-----------------------------------------------------------------
 * serialize detail::bs_npvec_impl
 *----------------------------------------------------------------*/
BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(save, blue_sky::detail::bs_npvec_impl, 1)
	typedef typename type::const_iterator citerator;
	typedef typename type::size_type size_t;

	// save array dims
	size_t ndim = t.ndim();
	ar << ndim;
	ar << boost::serialization::make_array(t.dims(), ndim);
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(load, blue_sky::detail::bs_npvec_impl, 1)
	typedef typename type::iterator iterator;
	typedef typename type::size_type size_t;

	// load array dims
	size_t ndim;
	ar >> ndim;
	std::vector< npy_intp > dims(ndim);
	ar >> boost::serialization::make_array(&dims[0], ndim);
	t.init(ndim, &dims[0]);
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_CLASS_SERIALIZE_SPLIT_T(blue_sky::detail::bs_npvec_impl, 1)

/*-----------------------------------------------------------------
 * serialize bs_npvec & bs_npvec_shared
 *----------------------------------------------------------------*/
BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(serialize, blue_sky::bs_npvec, 1)
	// just serialize impl
	ar & boost::serialization::bs_base_object< typename type::base_t, type >(t);
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(serialize, blue_sky::bs_npvec_shared, 1)
	// just serialize impl
	ar & boost::serialization::bs_base_object< typename type::base_t, type >(t);
BLUE_SKY_CLASS_SRZ_FCN_END

#endif /* end of include guard: BS_NPVEC_SERIALIZE_DUQHKGZW */

