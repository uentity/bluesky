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

BSS_FCN_DECL(serialize, blue_sky::objbase)

NAMESPACE_BEGIN(blue_sky)

template<typename Archive>
auto serialize(Archive& ar, error::box& t) -> void {
	ar(
		cereal::make_nvp("code", t.ec),
		cereal::make_nvp("domain", t.domain),
		cereal::make_nvp("message", t.message)
	);
}

NAMESPACE_END(blue_sky)

BSS_FORCE_DYNAMIC_INIT(base_types)
