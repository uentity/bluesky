/// @file
/// @author uentity
/// @date 25.03.2019
/// @brief Python binding for BS `property`
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/chrono.h>
#include <pybind11/stl.h>

#include "../property.h"
#include "../propdict.h"

// [NOTE] making propbooks transparent is much more convinient, but involves making data copies
// BUT properties aren't dedicated to hold large amount of data
// downside is that transaprent types can't be modified inplace (because of copies)
// here's a switch that allows to flip on using opaque propbooks exported by kernel
#ifdef BSPY_OPAQUE_PROPBOOK
	PYBIND11_MAKE_OPAQUE(blue_sky::prop::propbook_s);
	PYBIND11_MAKE_OPAQUE(blue_sky::prop::propbook_i);
#endif

NAMESPACE_BEGIN(pybind11::detail)

template<>
struct type_caster<blue_sky::prop::property> {
	using Type = blue_sky::prop::property;
	using UType = blue_sky::prop::property::underlying_type;
	PYBIND11_TYPE_CASTER(Type, _("property"));

	bool load(handle src, bool convert) {
		if(src.is_none()) {
			value = blue_sky::prop::none();
			return true;
		}
		auto caster = make_caster<UType>();
		if (caster.load(src, convert)) {
			value = cast_op<UType>(caster);
			return true;
		}
		return false;
	}

	template <typename Variant>
	static handle cast(Variant &&src, return_value_policy policy, handle parent) {
		return blue_sky::prop::is_none(src) ?
			pybind11::none().release() :
			make_caster<UType>::cast(std::forward<Variant>(src), policy, parent);
	}
};

NAMESPACE_END(pybind11::detail)
