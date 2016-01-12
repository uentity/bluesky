/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Include this in order to implement serialization support
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_SERIALIZE_MIZAXRNW
#define BS_SERIALIZE_MIZAXRNW

#include "bs_kernel.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/archive/polymorphic_iarchive.hpp>
#include <boost/archive/polymorphic_oarchive.hpp>

#include "bs_serialize_decl.h"
#include "bs_serialize_macro.h"
#include "bs_serialize_overl.h"
#include "bs_serialize_text.h"
#include "bs_serialize_fix.h"

#include "smart_ptr_serialize.h"
#include "bs_array_serialize.h"

// add empty serialize fcn for objbase
#include "bs_object_base.h"
//BLUE_SKY_CLASS_SRZ_FCN_DECL(serialize, blue_sky::objbase)

BLUE_SKY_TYPE_SERIALIZE_GUID(blue_sky::objbase)

#endif /* end of include guard: BS_SERIALIZE_MIZAXRNW */

