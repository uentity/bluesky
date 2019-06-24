/// @file
/// @author uentity
/// @date 04.06.2018
/// @brief Declare serialization of base BS types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../objbase.h"

#include "serialize_decl.h"
#include "carray.h"

NAMESPACE_BEGIN(blue_sky)
/*-----------------------------------------------------------------------------
 *  objbase
 *-----------------------------------------------------------------------------*/
BSS_FCN_DECL(serialize, blue_sky::objbase)

NAMESPACE_END(blue_sky)

BSS_FORCE_DYNAMIC_INIT(base_types)

