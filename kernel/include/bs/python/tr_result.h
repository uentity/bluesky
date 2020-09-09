/// @author uentity
/// @date 09.09.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../meta.h"
#include "../transaction.h"

#include "property.h"

NAMESPACE_BEGIN(pybind11::detail)

template<>
struct type_caster<blue_sky::tr_result> {
	using Type = blue_sky::tr_result;
	using UType = blue_sky::tr_result::underlying_type;
	PYBIND11_TYPE_CASTER(Type, _("tr_result"));

	bool load(handle src, bool convert) {
		using namespace blue_sky;

		if(src.is_none()) {
			value.emplace<1>(perfect);
			return true;
		}
		// first try to extract props
		auto caster = make_caster<prop::propdict>();
		if(caster.load(src, convert)) {
			value.emplace<0>(cast_op<prop::propdict>(std::move(caster)));
			return true;
		}
		// then E
		else {
			auto caster = make_caster<error>();
			if(caster.load(src, convert)) {
				value.emplace<1>(cast_op<error>(std::move(caster)));
				return true;
			}
		}
		return false;
	}

	template <typename Variant>
	static handle cast(Variant &&src, return_value_policy policy, handle parent) {
		return make_caster<UType>::cast(std::forward<Variant>(src), policy, parent);
	}
};

NAMESPACE_END(pybind11::detail)
