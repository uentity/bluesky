/// @file
/// @author uentity
/// @date 15.03.2017
/// @brief Include this in order to implement serialization support
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "../kernel.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/archive/polymorphic_iarchive.hpp>
#include <boost/archive/polymorphic_oarchive.hpp>

#include "serialize/decl.h"
#include "serialize/macro.h"
#include "serialize/text.h"
#include "serialize/fix.h"

#include "serialize/array_serialize.h"

// add empty serialize fcn for objbase
#include "../objbase.h"
//BLUE_SKY_CLASS_SRZ_FCN_DECL(serialize, blue_sky::objbase)

BLUE_SKY_TYPE_SERIALIZE_GUID(blue_sky::objbase)

