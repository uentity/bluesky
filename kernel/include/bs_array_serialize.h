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

#ifndef BS_ARRAY_SERIALIZE_ATPO4NI3
#define BS_ARRAY_SERIALIZE_ATPO4NI3

#include "bs_array.h"
#include <boost/serialization/serialization.hpp>
#include <boost/archive/polymorphic_iarchive.hpp>
#include <boost/archive/polymorphic_oarchive.hpp>

#define BS_ARRAY_FACTORY_PTR(T, cont_traits)                                    \
namespace boost {                                                               \
namespace archive { namespace detail {                                          \
template< >                                                                     \
struct heap_allocator< blue_sky::bs_array< T, blue_sky::cont_traits > > {       \
    typedef blue_sky::bs_array< T, blue_sky::cont_traits > type;                \
    typedef blue_sky::smart_ptr< type, true > sp_type;                          \
    static type* invoke() {                                                     \
        sp_type t = BS_KERNEL.create_object(type::bs_type(), false);            \
        return t.lock();                                                        \
    }                                                                           \
};                                                                              \
}} namespace serialization {                                                    \
template< >                                                                     \
void access::construct(blue_sky::bs_array< T, blue_sky::cont_traits >*) {}      \
template< >                                                                     \
void access::destroy(const blue_sky::bs_array< T, blue_sky::cont_traits >* t) { \
    typedef blue_sky::bs_array< T, blue_sky::cont_traits > type;                \
    typedef blue_sky::smart_ptr< type > sp_type;                                \
    BS_KERNEL.free_instance(sp_type(t));                                        \
}                                                                               \
}}


//namespace boost {
//namespace archive {
//
//class polymorphic_iarchive;
//class polymorphic_oarchive;
//
//} // namespace archive
//} // namespace boost

namespace boost { namespace serialization {

template< class Archive, class T, template< class > class cont_traits >
BS_API void serialize(
	Archive & ar,
	blue_sky::bs_array< T, cont_traits >& data,
	const unsigned int version
);

template< class Archive, class T, template< class > class cont_traits >
BS_API void load_construct_data(
	Archive & ar,
	blue_sky::bs_array< T, cont_traits >* data,
	const unsigned int version
);

}} /* boost::serialization */

namespace boost { namespace archive { namespace detail {

template< class T, template< class > class cont_traits >
struct BS_API heap_allocator< blue_sky::bs_array< T, cont_traits > > {
	typedef blue_sky::bs_array< T, cont_traits > type;
	typedef blue_sky::smart_ptr< type, true > sp_type;

	static type* invoke();
};

}}}


//BS_ARRAY_FACTORY_PTR(int, vector_traits)
//BS_ARRAY_FACTORY_PTR(unsigned int, vector_traits)
//BS_ARRAY_FACTORY_PTR(long, vector_traits)
//BS_ARRAY_FACTORY_PTR(unsigned long, vector_traits)
//BS_ARRAY_FACTORY_PTR(float, vector_traits)
//BS_ARRAY_FACTORY_PTR(double, vector_traits)
//BS_ARRAY_FACTORY_PTR(std::string, vector_traits)
//
//BS_ARRAY_FACTORY_PTR(int, bs_array_shared)
//BS_ARRAY_FACTORY_PTR(unsigned int, bs_array_shared)
//BS_ARRAY_FACTORY_PTR(long, bs_array_shared)
//BS_ARRAY_FACTORY_PTR(unsigned long, bs_array_shared)
//BS_ARRAY_FACTORY_PTR(float, bs_array_shared)
//BS_ARRAY_FACTORY_PTR(double, bs_array_shared)
//BS_ARRAY_FACTORY_PTR(std::string, bs_array_shared)
//
//BS_ARRAY_FACTORY_PTR(int, bs_vector_shared)
//BS_ARRAY_FACTORY_PTR(unsigned int, bs_vector_shared)
//BS_ARRAY_FACTORY_PTR(long, bs_vector_shared)
//BS_ARRAY_FACTORY_PTR(unsigned long, bs_vector_shared)
//BS_ARRAY_FACTORY_PTR(float, bs_vector_shared)
//BS_ARRAY_FACTORY_PTR(double, bs_vector_shared)
//BS_ARRAY_FACTORY_PTR(std::string, bs_vector_shared)

#endif /* end of include guard: BS_ARRAY_SERIALIZE_ATPO4NI3 */

