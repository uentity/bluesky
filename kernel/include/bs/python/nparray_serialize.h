/// @file
/// @author uentity
/// @date 15.03.2017
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "nparray.h"

#include "../compat/serialize.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/array.hpp>
#include <boost/archive/polymorphic_iarchive.hpp>
#include <boost/archive/polymorphic_oarchive.hpp>

/*-----------------------------------------------------------------
 * serialize bs_nparray_traits
 *----------------------------------------------------------------*/
BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(save, blue_sky::bs_nparray_traits, 1)
	typedef typename type::const_iterator citerator;
	typedef typename type::size_type size_t;

	// save array dims
	auto ndim = t.ndim();
	ar << ndim;
	ar << boost::serialization::make_array(t.shape(), ndim);
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(load, blue_sky::bs_nparray_traits, 1)
	typedef typename type::iterator iterator;
	typedef typename type::size_type size_t;

	// load array dims
	size_t ndim;
	ar >> ndim;
	std::vector< size_t > dims(ndim);
	ar >> boost::serialization::make_array(&dims[0], ndim);
	blue_sky::bs_nparray_traits< A0 >(dims[0], nullptr).swap(t);
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_CLASS_SERIALIZE_SPLIT_T(blue_sky::bs_nparray_traits, 1)

