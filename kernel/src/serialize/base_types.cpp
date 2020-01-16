/// @file
/// @author uentity
/// @date 04.06.2018
/// @brief Implement serialization of base BS types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include <bs/log.h>

#include <bs/serialize/serialize.h>
#ifdef BSPY_EXPORTING
#include <bs/python/common.h>
#include <bs/serialize/python.h>
#endif

#include <fmt/format.h>
#include <sstream>

using namespace cereal;

/*-----------------------------------------------------------------------------
 *  objbase
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, blue_sky::objbase)
	// dump typeid for text archives
	if constexpr(cereal::traits::is_text_archive<Archive>()) {
		if constexpr(typename Archive::is_saving()) {
			ar(make_nvp("typeid", const_cast<std::string&>(t.bs_resolve_type().name)));
		}
		else {
			std::string stype;
			ar(make_nvp("typeid", stype));
		}
	}

	ar(
		make_nvp("id", t.id_),
		make_nvp("is_node", t.is_node_)
	);
BSS_FCN_END

BSS_REGISTER_TYPE(blue_sky::objbase)
BSS_FCN_EXPORT(serialize, blue_sky::objbase)

BSS_REGISTER_DYNAMIC_INIT(base_types)
