/// @file
/// @author uentity
/// @date 28.06.2018
/// @brief Implementation of bs_array serialization
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include <bs/log.h>
#include <fmt/format.h>

#include <bs/serialize/array.h>
#include <bs/serialize/base_types.h>
#include <cereal/types/vector.hpp>

using namespace cereal;

NAMESPACE_BEGIN(blue_sky)

// check if archive supports binary serialization
template<typename Archive, typename T> using support_binary =
	cereal::traits::is_output_serializable<cereal::BinaryData<T>, Archive>;

/*-----------------------------------------------------------------------------
 *  bs_array
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN_EXT(serialize, bs_array, (class, template< class > class))
	// serialize via base classes
	ar(
		make_nvp("objbase", base_class<objbase>(&t))
	);
	ar(
		make_nvp("container", base_class<typename type::base_t>(&t))
	);
BSS_FCN_END

///////////////////////////////////////////////////////////////////////////////
// serialization functions for simple traits
//
BSS_FCN_INL_BEGIN_T(serialize, bs_arrbase_shared_impl, 2)
	ar(make_nvp("shared_data", t.shared_data_));
BSS_FCN_INL_END_T(serialize, bs_arrbase_shared_impl, 2)

BSS_FCN_INL_BEGIN_T(serialize, bs_arrbase_impl, 2)
	ar(base_class<typename type::container>(&t));
BSS_FCN_INL_END_T(serialize, bs_arrbase_impl, 2)

BSS_FCN_INL_BEGIN_T(serialize, bs_vector_shared, 1)
	ar(make_nvp("shared_data", t.shared_data_));
BSS_FCN_INL_END_T(serialize, bs_vector_shared, 1)

BSS_FCN_INL_BEGIN_T(serialize, vector_traits, 1)
	ar(base_class<typename type::base_t>(&t));
BSS_FCN_INL_END_T(serialize, vector_traits, 1)

/////////////////////////////////////////////////////////////////////////////
//  serialization of nparray traits
//
#ifdef BSPY_EXPORTING
BSS_FCN_INL_BEGIN_T(save, blue_sky::bs_nparray_traits, 1)
	// save array shape
	ar(make_nvp("shape", make_carray_view(t.shape(), t.ndim())));
	// save array data
	ar(make_nvp("values", make_carray_view(t.data(), t.size())));
BSS_FCN_INL_END_T(serialize, blue_sky::bs_nparray_traits, 1)

BSS_FCN_INL_BEGIN_T(load, blue_sky::bs_nparray_traits, 1)
	// load array shape
	std::vector< ssize_t > shape;
	ar(make_nvp("shape", shape));
	// create numpy array with given shape
	type(shape[0], nullptr).swap(t);
	// load array data
	ar(make_nvp("values", make_carray_view(t.data(), t.size())));
BSS_FCN_INL_END_T(serialize, blue_sky::bs_nparray_traits, 1)
#endif

NAMESPACE_END(blue_sky)

///////////////////////////////////////////////////////////////////////////////
//  instantiate & export serialization code
//
#define BSS_EXPORT_ARRAY(T, cont_traits)                                 \
BSS_REGISTER_TYPE_EXT(blue_sky::bs_array, (T, blue_sky::cont_traits))    \
BSS_FCN_EXPORT_EXT(serialize, blue_sky::bs_array, (T, blue_sky::cont_traits))

BSS_EXPORT_ARRAY(int                , vector_traits)
BSS_EXPORT_ARRAY(unsigned int       , vector_traits)
BSS_EXPORT_ARRAY(long               , vector_traits)
BSS_EXPORT_ARRAY(long long          , vector_traits)
BSS_EXPORT_ARRAY(unsigned long      , vector_traits)
BSS_EXPORT_ARRAY(unsigned long long , vector_traits)
BSS_EXPORT_ARRAY(float              , vector_traits)
BSS_EXPORT_ARRAY(double             , vector_traits)
BSS_EXPORT_ARRAY(std::string        , vector_traits)

BSS_EXPORT_ARRAY(int                , bs_vector_shared)
BSS_EXPORT_ARRAY(unsigned int       , bs_vector_shared)
BSS_EXPORT_ARRAY(long               , bs_vector_shared)
BSS_EXPORT_ARRAY(long long          , bs_vector_shared)
BSS_EXPORT_ARRAY(unsigned long      , bs_vector_shared)
BSS_EXPORT_ARRAY(unsigned long long , bs_vector_shared)
BSS_EXPORT_ARRAY(float              , bs_vector_shared)
BSS_EXPORT_ARRAY(double             , bs_vector_shared)
BSS_EXPORT_ARRAY(std::string        , bs_vector_shared)

#if defined(BSPY_EXPORTING)
BSS_EXPORT_ARRAY(int                , bs_nparray_traits)
BSS_EXPORT_ARRAY(unsigned int       , bs_nparray_traits)
BSS_EXPORT_ARRAY(long               , bs_nparray_traits)
BSS_EXPORT_ARRAY(long long          , bs_nparray_traits)
BSS_EXPORT_ARRAY(unsigned long      , bs_nparray_traits)
BSS_EXPORT_ARRAY(unsigned long long , bs_nparray_traits)
BSS_EXPORT_ARRAY(float              , bs_nparray_traits)
BSS_EXPORT_ARRAY(double             , bs_nparray_traits)
#endif

BSS_REGISTER_DYNAMIC_INIT(bs_array)

