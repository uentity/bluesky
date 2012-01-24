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

#if defined(BSPY_EXPORTING) && defined(UNIX)
// supress gcc warnings
#include <boost/python/detail/wrap_python.hpp>
#endif

#include "bs_serialize.h"

using namespace blue_sky;
namespace boser = boost::serialization;

BLUE_SKY_CLASS_SRZ_FCN_BEGIN(serialize, objbase)
	// register conversions to base classes
	// to omit their serialization
	boser::void_cast_register(
		static_cast< objbase* >(NULL),
		static_cast< bs_refcounter* >(NULL)
	);
	boser::void_cast_register(
		static_cast< objbase* >(NULL),
		static_cast< bs_messaging * >(NULL)
	);
	boser::void_cast_register(
		static_cast< bs_messaging* >(NULL),
		static_cast< bs_imessaging* >(NULL)
	);
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_TYPE_SERIALIZE_EXPORT(objbase)

