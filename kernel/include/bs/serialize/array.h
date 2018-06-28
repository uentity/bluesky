/// @file
/// @author uentity
/// @date 28.06.2018
/// @brief Declare serialization of bs_array
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "../compat/array.h"
#if defined(BSPY_EXPORTING) || defined(BSPY_EXPORTING_PLUGIN)
#include "../python/nparray.h"
#endif

#include "atomizer.h"

// bs_array
BSS_FCN_DECL_EXT(serialize, blue_sky::bs_array, (class, template< class > class))

BSS_FORCE_DYNAMIC_INIT(bs_array)

