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

#ifndef BS_SERIALIZE_TEXT_MMP1RZ0U
#define BS_SERIALIZE_TEXT_MMP1RZ0U

#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <sstream>

namespace blue_sky {

/*-----------------------------------------------------------------
 * helper functions that serialize any BS type to/from str
 *----------------------------------------------------------------*/

template< class T >
BS_API_PLUGIN std::string serialize_to_str(smart_ptr< T, true >& t) {
	std::ostringstream os;
	boost::archive::text_oarchive ar(os);
	ar << t;
	return os.str();
}

template< class T >
BS_API_PLUGIN smart_ptr< T, true > serialize_from_str(const std::string& src) {
	std::istringstream is(src);
	boost::archive::text_iarchive ar(is);
	smart_ptr< T, true > t;
	ar >> t;
	return t;
}

template< class T >
BS_API_PLUGIN smart_ptr< T, true > copy_via_serialize(smart_ptr< T, true > src) {
	return serialize_from_str< T >(serialize_to_str< T >(src));
}

}  // eof blue_sky namespace

#endif /* end of include guard: BS_SERIALIZE_MISC_MMP1RZ0U */

