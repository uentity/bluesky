/// @file
/// @author uentity
/// @date 21.04.2020
/// @brief Transparent C++ UUID <-> Python conversion
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include <bs/uuid.h>

NAMESPACE_BEGIN(pybind11::detail)

template<>
struct type_caster<blue_sky::uuid> {
	using uuid_t = blue_sky::uuid;
	PYBIND11_TYPE_CASTER(uuid_t, _("bsuuid"));

	static auto py_uuid() -> object& {
		static object type_ = module::import("uuid").attr("UUID");
		return type_;
	}

	auto load(handle src, bool) -> bool {
		if(isinstance(src, py_uuid())) {
			auto buf = static_cast<std::string>(reinterpret_borrow<bytes>(src.attr("bytes")));
			std::copy(buf.begin(), buf.end(), value.begin());
			return true;
		}
		return false;
	}

	static handle cast(uuid_t src, return_value_policy /* policy */, handle /* parent */) {
		auto buf = bytes(reinterpret_cast<const char*>(src.data), src.size());
		return py_uuid()("bytes"_a = std::move(buf)).release();
	}
};

NAMESPACE_END(pybind11::detail)
