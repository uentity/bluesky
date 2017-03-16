/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/compat/serialize.h>
#include <bs/compat/array_serialize.h>
#ifdef BSPY_EXPORTING
#include <bs/python/nparray_serialize.h>
#endif

#include <boost/serialization/export.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/string.hpp>
// instantiate code for text archives for compatibility reason
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#define BS_ARRAY_EXPORT(T, cont_traits) \
BLUE_SKY_TYPE_SERIALIZE_IMPL_EXT(blue_sky::bs_array, 2, (T, blue_sky::cont_traits))

using namespace blue_sky;
namespace boser = boost::serialization;

/*-----------------------------------------------------------------
 * serialize bs_array
 *----------------------------------------------------------------*/
BLUE_SKY_CLASS_SRZ_FCN_BEGIN_EXT(save, bs_array, 2, (class, template< class > class))
	typedef typename type::const_iterator citerator;
	// save array size
	const ulong sz = t.size();
	ar << sz;
	// save data
	for(citerator pd = t.begin(), end = t.end(); pd != end; ++pd)
		ar << *pd;
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_CLASS_SRZ_FCN_BEGIN_EXT(load, bs_array, 2, (class, template< class > class))
	typedef typename type::iterator iterator;
	// restore size
	ulong sz;
	ar >> sz;
	t.resize(sz);
	// restore data
	for(iterator pd = t.begin(), end = t.end(); pd != end; ++pd)
		ar >> *pd;
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_CLASS_SRZ_FCN_BEGIN_EXT(serialize, blue_sky::bs_array, 2, (class, template< class > class))
	// invoke serialization of base class first
	typedef typename type::base_t base_t;
	ar & boser::base_object< base_t, type >(t);

	// split
	boser::split_free(ar, t, version);
BLUE_SKY_CLASS_SRZ_FCN_END

/*-----------------------------------------------------------------
 * serializate array traits
 *----------------------------------------------------------------*/
////////////////////////////////////////////////////////////////////
// empty serialization functions for simple traits
//
BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(serialize, bs_vecbase_impl, 2)
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(serialize, bs_array_shared, 1)
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(serialize, bs_vector_shared, 1)
BLUE_SKY_CLASS_SRZ_FCN_END

/*-----------------------------------------------------------------
 * instantiate serialization code
 *----------------------------------------------------------------*/
BS_ARRAY_EXPORT(int, vector_traits)
BS_ARRAY_EXPORT(unsigned int, vector_traits)
BS_ARRAY_EXPORT(long, vector_traits)
BS_ARRAY_EXPORT(long long, vector_traits)
BS_ARRAY_EXPORT(unsigned long, vector_traits)
BS_ARRAY_EXPORT(unsigned long long, vector_traits)
BS_ARRAY_EXPORT(float, vector_traits)
BS_ARRAY_EXPORT(double, vector_traits)
BS_ARRAY_EXPORT(std::string, vector_traits)
BS_ARRAY_EXPORT(std::wstring, vector_traits)

//BS_ARRAY_EXPORT(int, bs_array_shared)
//BS_ARRAY_EXPORT(unsigned int, bs_array_shared)
//BS_ARRAY_EXPORT(long, bs_array_shared)
//BS_ARRAY_EXPORT(long long, bs_array_shared)
//BS_ARRAY_EXPORT(unsigned long, bs_array_shared)
//BS_ARRAY_EXPORT(unsigned long long, bs_array_shared)
//BS_ARRAY_EXPORT(float, bs_array_shared)
//BS_ARRAY_EXPORT(double, bs_array_shared)
//BS_ARRAY_EXPORT(std::string, bs_array_shared)
//BS_ARRAY_EXPORT(std::wstring, bs_array_shared)

BS_ARRAY_EXPORT(int, bs_vector_shared)
BS_ARRAY_EXPORT(unsigned int, bs_vector_shared)
BS_ARRAY_EXPORT(long, bs_vector_shared)
BS_ARRAY_EXPORT(long long, bs_vector_shared)
BS_ARRAY_EXPORT(unsigned long, bs_vector_shared)
BS_ARRAY_EXPORT(unsigned long long, bs_vector_shared)
BS_ARRAY_EXPORT(float, bs_vector_shared)
BS_ARRAY_EXPORT(double, bs_vector_shared)
BS_ARRAY_EXPORT(std::string, bs_vector_shared)
BS_ARRAY_EXPORT(std::wstring, bs_vector_shared)

#ifdef BSPY_EXPORTING
BS_ARRAY_EXPORT(int                , bs_nparray_traits)
BS_ARRAY_EXPORT(unsigned int       , bs_nparray_traits)
BS_ARRAY_EXPORT(long               , bs_nparray_traits)
BS_ARRAY_EXPORT(long long          , bs_nparray_traits)
BS_ARRAY_EXPORT(unsigned long      , bs_nparray_traits)
BS_ARRAY_EXPORT(unsigned long long , bs_nparray_traits)
BS_ARRAY_EXPORT(float              , bs_nparray_traits)
BS_ARRAY_EXPORT(double             , bs_nparray_traits)
#endif
