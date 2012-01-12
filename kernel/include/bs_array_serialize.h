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
inline void serialize(
	Archive & ar, bs_array< T, cont_traits >& data, const unsigned int version
)
{
	split_free(ar, data, version);
}

}} /* boost::serialization */

#endif /* end of include guard: BS_ARRAY_SERIALIZE_ATPO4NI3 */

