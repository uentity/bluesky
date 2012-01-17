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

//#include <boost/archive/text_iarchive.hpp>
//#include <boost/archive/text_oarchive.hpp>

#include <boost/serialization/export.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/string.hpp>

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

#define BS_ARRAY_EXPORT(T, cont_traits)     \
BLUE_SKY_TYPE_SERIALIZE_GUID_EXT(blue_sky::bs_array, 2, (T, blue_sky::cont_traits)) \
BLUE_SKY_TYPE_SERIALIZE_EXPORT_EXT(blue_sky::bs_array, 2, (T, blue_sky::cont_traits))

namespace boost { namespace serialization {
using namespace blue_sky;

template< class Archive, class T, template< class > class cont_traits >
void save(Archive& ar, const bs_array< T, cont_traits >& data, const unsigned int version) {
	typedef typename bs_array< T, cont_traits >::const_iterator citerator;
	// save array size
	ar << (const ulong&)data.size();
	// save data
	for(citerator pd = data.begin(), end = data.end(); pd != end; ++pd)
		ar << *pd;
}

template< class Archive, class T, template< class > class cont_traits >
void load(Archive& ar, bs_array< T, cont_traits >& data, const unsigned int version) {
	typedef typename bs_array< T, cont_traits >::iterator iterator;
	// restore size
	ulong sz;
	ar >> sz;
	data.resize(sz);
	// restore data
	for(iterator pd = data.begin(), end = data.end(); pd != end; ++pd)
		ar >> *pd;
}

// override serialize
template< class Archive, class T, template< class > class cont_traits >
void serialize(
	Archive & ar, bs_array< T, cont_traits >& data, const unsigned int version
){
	split_free(ar, data, version);
}

}} /* boost::serialization */

BLUE_SKY_TYPE_SERIALIZE_IMPL_EXT(blue_sky::bs_array, 2, (class, template< class > class))

BS_ARRAY_EXPORT(int, vector_traits)
BS_ARRAY_EXPORT(unsigned int, vector_traits)
BS_ARRAY_EXPORT(long, vector_traits)
BS_ARRAY_EXPORT(unsigned long, vector_traits)
BS_ARRAY_EXPORT(float, vector_traits)
BS_ARRAY_EXPORT(double, vector_traits)
BS_ARRAY_EXPORT(std::string, vector_traits)

BS_ARRAY_EXPORT(int, bs_array_shared)
BS_ARRAY_EXPORT(unsigned int, bs_array_shared)
BS_ARRAY_EXPORT(long, bs_array_shared)
BS_ARRAY_EXPORT(unsigned long, bs_array_shared)
BS_ARRAY_EXPORT(float, bs_array_shared)
BS_ARRAY_EXPORT(double, bs_array_shared)
BS_ARRAY_EXPORT(std::string, bs_array_shared)

BS_ARRAY_EXPORT(int, bs_vector_shared)
BS_ARRAY_EXPORT(unsigned int, bs_vector_shared)
BS_ARRAY_EXPORT(long, bs_vector_shared)
BS_ARRAY_EXPORT(unsigned long, bs_vector_shared)
BS_ARRAY_EXPORT(float, bs_vector_shared)
BS_ARRAY_EXPORT(double, bs_vector_shared)
BS_ARRAY_EXPORT(std::string, bs_vector_shared)

