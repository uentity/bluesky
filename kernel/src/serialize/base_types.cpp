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
#include <bs/serialize/base_types.h>
#ifdef BSPY_EXPORTING
#include <bs/python/common.h>
#include <bs/serialize/python.h>
#endif

#include <spdlog/fmt/fmt.h>
#include <sstream>

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

		// emit only in text archives
		static auto save_typeid(Archive& ar, const type& t, std::true_type) -> void {
			save_(ar, t, is_saving());
		}
		// and not in binary
		static auto save_typeid(Archive& ar, const type& t, std::false_type) -> void {}
	};

	if_saving::save_typeid(ar, t, cereal::traits::is_text_archive<Archive>());
	ar(
		make_nvp("id", t.id_),
		make_nvp("is_node", t.is_node_)
	);
BSS_FCN_END

BSS_REGISTER_TYPE(blue_sky::objbase)
BSS_FCN_EXPORT(serialize, blue_sky::objbase)

#ifdef BSPY_EXPORTING_PLUGIN
/*-----------------------------------------------------------------------------
 *  py_object
 *-----------------------------------------------------------------------------*/
using py_object = blue_sky::python::py_object<blue_sky::objbase>;

BSS_FCN_INL_BEGIN(serialize, py_object)
	ar(
		make_nvp("objbase", cereal::base_class<blue_sky::objbase>(&t)),
		make_nvp("pyobj", t.pyobj)
	);
BSS_FCN_INL_END(serialize, py_object)

// [NOTE] using `CEREAL_REGISTER_TYPE` because BSS_REGISTER_TYPE will extract type name from `objbase`
// and this will lead to conflict with `objbase` itself and loading errors
CEREAL_REGISTER_TYPE(py_object)
BSS_FCN_EXPORT(serialize, py_object)
#endif

BSS_REGISTER_DYNAMIC_INIT(base_types)

