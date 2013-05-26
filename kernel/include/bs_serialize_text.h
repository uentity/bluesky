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
#include <boost/serialization/split_free.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <sstream>

#include "bs_serialize_fix.h"

namespace blue_sky {

/*-----------------------------------------------------------------
 * helper functions that serialize any BS type to/from str
 *----------------------------------------------------------------*/

template< class T >
BS_API_PLUGIN std::string serialize_to_str(const smart_ptr< T, true >& t) {
	typedef typename boost::archive::text_oarchive archive_t;
	std::ostringstream os;
	// ensure archive is destroyed before stream is closed
	{
		archive_t ar(os);
		serialize_fix_data< archive_t >(ar) << t;
	}
	return os.str();
}

template< class T >
BS_API_PLUGIN smart_ptr< T, true > serialize_from_str(const std::string& src) {
	typedef typename boost::archive::text_iarchive archive_t;
	std::istringstream is(src);
	smart_ptr< T, true > t;
	// ensure archive is destroyed before stream is closed
	{
		archive_t ar(is);
		serialize_fix_data< archive_t >(ar) >> t;
	}
	return t;
}

template< class T >
BS_API_PLUGIN smart_ptr< T, true > copy_via_serialize(const smart_ptr< T, true >& src) {
	return serialize_from_str< T >(serialize_to_str< T >(src));
}

/*-----------------------------------------------------------------
 * indirect string serialization -- can be used with abstract interfaces
 *----------------------------------------------------------------*/
// 2-nd param in both functions used only to pass type information
// R is a temporary type used to actually invoke serialization sunctions inside

template< class T, class R >
BS_API_PLUGIN std::string serialize_to_str_indirect(
	const smart_ptr< T, true >& t)
{
	return serialize_to_str(smart_ptr< R, true >(t, bs_static_cast()));
}

template< class T, class R >
BS_API_PLUGIN smart_ptr< T, true > serialize_from_str_indirect(
	const std::string& src)
{
	return smart_ptr< T, true >(serialize_from_str< R >(src), bs_static_cast());
}

}  // eof blue_sky namespace

#endif /* end of include guard: BS_SERIALIZE_MISC_MMP1RZ0U */

