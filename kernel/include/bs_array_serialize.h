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
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/export.hpp>

namespace boost { namespace serialization {
using namespace blue_sky;

template< class Archive, class T, template< class > class cont_traits >
void save(Archive& ar, const smart_ptr< bs_array< T, cont_traits > >& data, const unsigned int version) {
	// save array size
	ar << (const ulong&)data->size();
	// save data
	for(ulong i = 0; i < data->size(); ++i)
		ar << data->ss(i);
}

template< class Archive, class T, template< class > class cont_traits >
void load(Archive& ar, smart_ptr< bs_array< T, cont_traits > >& data, const unsigned int version) {
	// create array
	if(!data)
		data = BS_KERNEL.create_object(bs_array< T, cont_traits >::bs_type());
	// restore size
	ulong sz;
	ar >> sz;
	data->resize(sz);
	// restore data
	for(ulong i = 0; i < data->size(); ++i)
		ar >> data->ss(i);
}

// override serialize
template< class Archive, class T, template< class > class cont_traits >
inline void serialize(
	Archive & ar, smart_ptr< bs_array< T, cont_traits > >& data, const unsigned int version
)
{
	split_free(ar, data, version);
}

// define guids for arrays
//template< class T, template< class > class cont_traits >
//struct guid_defined< smart_ptr< bs_array< T, cont_traits > > > : boost::mpl::true_ {};

//template< class T, template< class > class cont_traits >
//inline const char * guid< smart_ptr< bs_array< T, cont_traits > > >() {
//	return bs_array< T, cont_traits >::bs_type().stype_;
//}


}} /* boost::serialization */

#endif /* end of include guard: BS_ARRAY_SERIALIZE_ATPO4NI3 */

