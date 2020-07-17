/// @file
/// @author uentity
/// @date 04.06.2018
/// @brief Implement serialization of base BS types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel/radio.h>
#include <bs/serialize/base_types.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/serialize.h>

#include <caf/group.hpp>

using namespace cereal;

/*-----------------------------------------------------------------------------
 *  objbase
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, blue_sky::objbase)
	ar(make_nvp("id", t.id_));
	if constexpr(Archive::is_saving::value)
		ar(make_nvp("home_id", t.home_id()));
	else {
		// reed home ID & reset home group
		std::string hid;
		ar(make_nvp("home_id", hid));
		t.reset_home(std::move(hid), false);
	}
BSS_FCN_END

BSS_FCN_EXPORT(serialize, blue_sky::objbase)
BSS_REGISTER_TYPE(blue_sky::objbase)

/*-----------------------------------------------------------------------------
 *  objnode
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, blue_sky::objnode)
	ar(cereal::base_class<objbase>(&t));
	ar(make_nvp("objnode", t.node_));
BSS_FCN_END

BSS_REGISTER_TYPE(blue_sky::objnode)
BSS_FCN_EXPORT(serialize, blue_sky::objnode)

BSS_REGISTER_DYNAMIC_INIT(base_types)
