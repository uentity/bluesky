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

#define BS_ARRAY_EXPORT_IMPLEMENT(T, cont_traits)                                            \
namespace boost { namespace archive { namespace detail { namespace extra_detail {            \
template< >                                                                                  \
struct init_guid< ::blue_sky::bs_array< T, ::blue_sky::cont_traits > > {                     \
    static guid_initializer< ::blue_sky::bs_array< T, ::blue_sky::cont_traits > > const & g; \
};                                                                                           \
guid_initializer< ::blue_sky::bs_array< T, ::blue_sky::cont_traits > > const &               \
init_guid< ::blue_sky::bs_array < T, blue_sky::cont_traits > >::g =                          \
    ::boost::serialization::singleton<                                                       \
        guid_initializer< ::blue_sky::bs_array < T, ::blue_sky::cont_traits > >              \
    >::get_mutable_instance().export_guid();                                                 \
}}}}

#define BS_ARRAY_GUID_VALUE(T, cont_traits)                                            \
namespace boost { namespace serialization {                                            \
template< >                                                                            \
struct guid_defined<                                                                   \
    blue_sky::bs_array< T, ::blue_sky::cont_traits >                                   \
> : boost::mpl::true_ {};                                                              \
template< >                                                                            \
inline const char * guid< blue_sky::bs_array< T, ::blue_sky::cont_traits > >() {       \
    return blue_sky::bs_array< T, ::blue_sky::cont_traits >::bs_type().stype_.c_str(); \
}                                                                                      \
}}

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

#define BS_ARRAY_GUID(T, cont_traits)     \
BS_ARRAY_GUID_VALUE(T, cont_traits)       \
BS_ARRAY_EXPORT_IMPLEMENT(T, cont_traits) \
//BS_ARRAY_FACTORY_PTR(T, cont_traits)

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

// default construct does nothing
template< class Archive, class T, template< class > class cont_traits >
void load_construct_data(
	Archive & ar,
	blue_sky::bs_array< T, cont_traits >* data,
	const unsigned int version
)
{}

}} /* boost::serialization */

namespace boost { namespace archive { namespace detail {

template< class T, template< class > class cont_traits >
typename heap_allocator< blue_sky::bs_array< T, cont_traits > >::type*
heap_allocator< blue_sky::bs_array< T, cont_traits > >::invoke() {
	sp_type t = BS_KERNEL.create_object(type::bs_type(), false);
	return t.lock();
}

}}} /* boost::archive::detail */

BS_ARRAY_GUID(int, vector_traits)
BS_ARRAY_GUID(unsigned int, vector_traits)
BS_ARRAY_GUID(long, vector_traits)
BS_ARRAY_GUID(unsigned long, vector_traits)
BS_ARRAY_GUID(float, vector_traits)
BS_ARRAY_GUID(double, vector_traits)
BS_ARRAY_GUID(std::string, vector_traits)

BS_ARRAY_GUID(int, bs_array_shared)
BS_ARRAY_GUID(unsigned int, bs_array_shared)
BS_ARRAY_GUID(long, bs_array_shared)
BS_ARRAY_GUID(unsigned long, bs_array_shared)
BS_ARRAY_GUID(float, bs_array_shared)
BS_ARRAY_GUID(double, bs_array_shared)
BS_ARRAY_GUID(std::string, bs_array_shared)

BS_ARRAY_GUID(int, bs_vector_shared)
BS_ARRAY_GUID(unsigned int, bs_vector_shared)
BS_ARRAY_GUID(long, bs_vector_shared)
BS_ARRAY_GUID(unsigned long, bs_vector_shared)
BS_ARRAY_GUID(float, bs_vector_shared)
BS_ARRAY_GUID(double, bs_vector_shared)
BS_ARRAY_GUID(std::string, bs_vector_shared)

