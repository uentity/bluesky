/// @file
/// @author uentity
/// @date 22.04.2019
/// @brief Transparent `object_ptr` caster
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <pybind11/pybind11.h>
#include "../detail/object_ptr.h"

NAMESPACE_BEGIN(pybind11::detail)

template<typename T> struct type_caster<blue_sky::object_ptr<T>> {
	using Type = blue_sky::object_ptr<T>;
	PYBIND11_TYPE_CASTER(Type, _("object_ptr[") + make_caster<T>::name + _("]"));

	bool load(handle src, bool convert) {
		auto caster = make_caster<T>();
		if(caster.load(src, convert)) {
			value = cast_op<T*>(caster);
			return true;
		}
		return false;
	}

	static handle cast(Type ptr, return_value_policy policy, handle parent) {
		return make_caster<T>::cast(ptr.get(), policy, parent);
	}
};

NAMESPACE_END(pybind11::detail)
