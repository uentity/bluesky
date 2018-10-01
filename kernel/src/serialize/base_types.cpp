/// @file
/// @author uentity
/// @date 04.06.2018
/// @brief Implement serialization of base BS types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel.h>
#include <bs/error.h>
#include <bs/log.h>
#include <spdlog/fmt/fmt.h>
#include <sstream>

#include <bs/serialize/base_types.h>

using namespace cereal;

/*-----------------------------------------------------------------------------
 *  objbase
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, blue_sky::objbase)
	// store typeid, but don't read it
	struct if_saving {
		using is_saving = typename Archive::is_saving;

		static auto save_(Archive& ar, const type& t, std::true_type) -> void {
			ar(make_nvp("typeid", const_cast<std::string&>(t.bs_resolve_type().name)));
		}
		static auto save_(Archive& ar, const type& t, std::false_type) -> void {
			std::string stype;
			ar(make_nvp("typeid", stype));
			//if(stype != type::bs_type().name)
			//	throw error(fmt::format(
			//		"Trying to deserialize '{}` into object of type `{}`", stype, type::bs_type().name
			//	));
		}

		static auto save_typeid(Archive& ar, const type& t) -> void {
			save_(ar, t, is_saving());
		}
	};

	if_saving::save_typeid(ar, t);
	ar(
		make_nvp("id", t.id_),
		make_nvp("is_node", t.is_node_)
	);
BSS_FCN_END

BSS_REGISTER_TYPE(blue_sky::objbase)
BSS_FCN_EXPORT(serialize, blue_sky::objbase)

BSS_REGISTER_DYNAMIC_INIT(base_types)

