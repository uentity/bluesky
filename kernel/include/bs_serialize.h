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

#ifndef BS_SERIALIZE_MIZAXRNW
#define BS_SERIALIZE_MIZAXRNW

#include "bs_common.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/archive/polymorphic_iarchive.hpp>
#include <boost/archive/polymorphic_oarchive.hpp>
// instantiate code for text archives for compatibility reason
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "bs_serialize_macro.h"
#include "bs_serialize_decl.h"
#include "bs_serialize_overl.h"

#include "smart_ptr_serialize.h"
#include "bs_array_serialize.h"

// add empty serialize fcn for objbase
#include "bs_object_base.h"
BLUE_SKY_CLASS_SRZ_FCN_DECL(serialize, blue_sky::objbase)

BLUE_SKY_TYPE_SERIALIZE_GUID(blue_sky::objbase)

/*-----------------------------------------------------------------
 * helper functions that serialize any BS type to/from str
 *----------------------------------------------------------------*/

#include <sstream>

namespace blue_sky {

template< class T >
BS_API_PLUGIN std::string serialize_to_str(smart_ptr< T, true >& t) {
	std::ostringstream os;
	boost::archive::text_oarchive ar(os);
	ar << t;
	return os.str();
}

template< class T >
BS_API_PLUGIN smart_ptr< T, true > serialize_from_str(const std::string& src) {
	std::istringstream is;
	boost::archive::text_iarchive ar(is);
	smart_ptr< T, true > t;
	ar >> t;
	return t;
}

}  // eof blue_sky namespace

#endif /* end of include guard: BS_SERIALIZE_MIZAXRNW */

