/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_ARRAY_SERIALIZE_ATPO4NI3
#define BS_ARRAY_SERIALIZE_ATPO4NI3

#include "bs_array.h"
#include "bs_serialize_macro.h"
#include "bs_serialize_decl.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/archive/polymorphic_iarchive.hpp>
#include <boost/archive/polymorphic_oarchive.hpp>

#define BS_ARRAY_GUID_VALUE(T, cont_traits) \
BLUE_SKY_TYPE_SERIALIZE_GUID_EXT(blue_sky::bs_array, 2, (T, blue_sky::cont_traits))

BLUE_SKY_TYPE_SERIALIZE_DECL_EXT(blue_sky::bs_array, 2, (class, template< class > class))

BLUE_SKY_CLASS_SRZ_FCN_DECL_EXT(serialize, blue_sky::bs_array, 2, (class, template< class > class))

BS_ARRAY_GUID_VALUE(int, vector_traits)
BS_ARRAY_GUID_VALUE(unsigned int, vector_traits)
BS_ARRAY_GUID_VALUE(long, vector_traits)
BS_ARRAY_GUID_VALUE(long long, vector_traits)
BS_ARRAY_GUID_VALUE(unsigned long, vector_traits)
BS_ARRAY_GUID_VALUE(unsigned long long, vector_traits)
BS_ARRAY_GUID_VALUE(float, vector_traits)
BS_ARRAY_GUID_VALUE(double, vector_traits)
BS_ARRAY_GUID_VALUE(std::string, vector_traits)

BS_ARRAY_GUID_VALUE(int, bs_array_shared)
BS_ARRAY_GUID_VALUE(unsigned int, bs_array_shared)
BS_ARRAY_GUID_VALUE(long, bs_array_shared)
BS_ARRAY_GUID_VALUE(long long, bs_array_shared)
BS_ARRAY_GUID_VALUE(unsigned long, bs_array_shared)
BS_ARRAY_GUID_VALUE(unsigned long long, bs_array_shared)
BS_ARRAY_GUID_VALUE(float, bs_array_shared)
BS_ARRAY_GUID_VALUE(double, bs_array_shared)
BS_ARRAY_GUID_VALUE(std::string, bs_array_shared)

BS_ARRAY_GUID_VALUE(int, bs_vector_shared)
BS_ARRAY_GUID_VALUE(unsigned int, bs_vector_shared)
BS_ARRAY_GUID_VALUE(long, bs_vector_shared)
BS_ARRAY_GUID_VALUE(long long, bs_vector_shared)
BS_ARRAY_GUID_VALUE(unsigned long, bs_vector_shared)
BS_ARRAY_GUID_VALUE(unsigned long long, bs_vector_shared)
BS_ARRAY_GUID_VALUE(float, bs_vector_shared)
BS_ARRAY_GUID_VALUE(double, bs_vector_shared)
BS_ARRAY_GUID_VALUE(std::string, bs_vector_shared)

#if defined(BSPY_EXPORTING) || defined(BSPY_EXPORTING_PLUGIN)
#include "bs_npvec_shared.h"

BS_ARRAY_GUID_VALUE(int, bs_npvec)
BS_ARRAY_GUID_VALUE(unsigned int, bs_npvec)
BS_ARRAY_GUID_VALUE(long, bs_npvec)
BS_ARRAY_GUID_VALUE(long long, bs_npvec)
BS_ARRAY_GUID_VALUE(unsigned long, bs_npvec)
BS_ARRAY_GUID_VALUE(unsigned long long, bs_npvec)
BS_ARRAY_GUID_VALUE(float, bs_npvec)
BS_ARRAY_GUID_VALUE(double, bs_npvec)
BS_ARRAY_GUID_VALUE(std::string, bs_npvec)

BS_ARRAY_GUID_VALUE(int, bs_npvec_shared)
BS_ARRAY_GUID_VALUE(unsigned int, bs_npvec_shared)
BS_ARRAY_GUID_VALUE(long, bs_npvec_shared)
BS_ARRAY_GUID_VALUE(long long, bs_npvec_shared)
BS_ARRAY_GUID_VALUE(unsigned long, bs_npvec_shared)
BS_ARRAY_GUID_VALUE(unsigned long long, bs_npvec_shared)
BS_ARRAY_GUID_VALUE(float, bs_npvec_shared)
BS_ARRAY_GUID_VALUE(double, bs_npvec_shared)
BS_ARRAY_GUID_VALUE(std::string, bs_npvec_shared)
#endif

#endif /* end of include guard: BS_ARRAY_SERIALIZE_ATPO4NI3 */

