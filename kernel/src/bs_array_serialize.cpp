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

#include "bs_array_serialize.h"
#include "bs_serialize_overl.h"
#include "bs_serialize_fixreal.h"

#include <boost/serialization/export.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/string.hpp>
// instantiate code for text archives for compatibility reason
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>


#define BS_ARRAY_FACTORY_N(T, cont_traits, N)                                \
namespace boost { namespace serialization {                                  \
template< >                                                                  \
blue_sky::bs_array< T, blue_sky::cont_traits >*                              \
factory< blue_sky::bs_array< T, blue_sky::cont_traits >, N >(std::va_list) { \
    return static_cast< blue_sky::bs_array< T, blue_sky::cont_traits >* >(   \
        blue_sky::bs_array< T, blue_sky::cont_traits >::bs_create_instance() \
    );                                                                       \
}                                                                            \
}}

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
	ar & boser::bs_base_object< base_t, type >(t);

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

BS_ARRAY_EXPORT(int, bs_array_shared)
BS_ARRAY_EXPORT(unsigned int, bs_array_shared)
BS_ARRAY_EXPORT(long, bs_array_shared)
BS_ARRAY_EXPORT(long long, bs_array_shared)
BS_ARRAY_EXPORT(unsigned long, bs_array_shared)
BS_ARRAY_EXPORT(unsigned long long, bs_array_shared)
BS_ARRAY_EXPORT(float, bs_array_shared)
BS_ARRAY_EXPORT(double, bs_array_shared)
BS_ARRAY_EXPORT(std::string, bs_array_shared)
BS_ARRAY_EXPORT(std::wstring, bs_array_shared)

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

#if defined(BSPY_EXPORTING) || defined(BSPY_EXPORTING_PLUGIN)
#include "bs_npvec_serialize.h"

BS_ARRAY_EXPORT(int, bs_npvec)
BS_ARRAY_EXPORT(unsigned int, bs_npvec)
BS_ARRAY_EXPORT(long, bs_npvec)
BS_ARRAY_EXPORT(long long, bs_npvec)
BS_ARRAY_EXPORT(unsigned long, bs_npvec)
BS_ARRAY_EXPORT(unsigned long long, bs_npvec)
BS_ARRAY_EXPORT(float, bs_npvec)
BS_ARRAY_EXPORT(double, bs_npvec)
BS_ARRAY_EXPORT(std::string, bs_npvec)
BS_ARRAY_EXPORT(std::wstring, bs_npvec)

BS_ARRAY_EXPORT(int, bs_npvec_shared)
BS_ARRAY_EXPORT(unsigned int, bs_npvec_shared)
BS_ARRAY_EXPORT(long, bs_npvec_shared)
BS_ARRAY_EXPORT(long long, bs_npvec_shared)
BS_ARRAY_EXPORT(unsigned long, bs_npvec_shared)
BS_ARRAY_EXPORT(unsigned long long, bs_npvec_shared)
BS_ARRAY_EXPORT(float, bs_npvec_shared)
BS_ARRAY_EXPORT(double, bs_npvec_shared)
BS_ARRAY_EXPORT(std::string, bs_npvec_shared)
BS_ARRAY_EXPORT(std::wstring, bs_npvec_shared)
#endif
