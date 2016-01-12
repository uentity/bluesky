/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

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
	boser::bs_void_cast_register(
		static_cast< objbase* >(NULL),
		static_cast< bs_messaging * >(NULL)
	);
	boser::void_cast_register(
		static_cast< bs_messaging* >(NULL),
		static_cast< bs_imessaging* >(NULL)
	);
BLUE_SKY_CLASS_SRZ_FCN_END

BLUE_SKY_TYPE_SERIALIZE_IMPL(objbase)

