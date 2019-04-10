/// @file
/// @author uentity
/// @date 26.09.2017
/// @brief Python binding for std::any
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/chrono.h>
#include "../fwd.h"
#include "../timetypes.h"

#include <any>

#define bs_any std::any

NAMESPACE_BEGIN(pybind11::detail)
/*-----------------------------------------------------------------------------
 *  cast blue_sky::type_tuple <-> Python tuple
 *-----------------------------------------------------------------------------*/
template<typename... Ts>
struct any_caster {
	using Type = std::any;
	PYBIND11_TYPE_CASTER(Type, _("any[") + detail::concat(make_caster<Ts>::name...) + _("]"));

	static constexpr auto cast_seq_v = type_list<Ts...>{};

	///////////////////////////////////////////////////////////////////////////////
	//  Python -> C++
	//
	template <typename U, typename... Us>
	auto load_alternative(handle src, bool convert, type_list<U, Us...>) -> bool {
		namespace py = pybind11;
		// code is the same as `std::variant` caster from pybind11
		// but with two exceptions for quick `handle` or `object` passthrough
		if constexpr(std::is_same_v<U, py::object>) {
			value = reinterpret_borrow<py::object>(src);
		}
		else {
			auto caster = make_caster<U>();
			if(caster.load(src, convert)) {
				value = cast_op<U>(caster);
				return true;
			}
		}
		return load_alternative(src, convert, type_list<Us...>());
	}
	auto load_alternative(handle, bool, type_list<>) { return false; }

	bool load(handle src, bool convert) {
		// Do a first pass without conversions to improve constructor resolution.
		// E.g. `py::int_(1).cast<variant<double, int>>()` needs to fill the `int`
		// slot of the variant. Without two-pass loading `double` would be filled
		// because it appears first and a conversion is possible.
		if(convert && load_alternative(src, false, cast_seq_v))
			return true;
		return load_alternative(src, convert, cast_seq_v);
	}

	///////////////////////////////////////////////////////////////////////////////
	//  C++ -> Python
	//
	template<typename Any, typename U, typename... Us>
	static auto cast_alternative(
		Any* src, return_value_policy pol, handle parent, type_list<U, Us...>
	) -> handle {
		if(auto src_v = std::any_cast<U>(src)) {
			if constexpr(std::is_same_v<U, object>)
				return *src_v;
			else
				return make_caster<U>().cast(*src_v, pol, parent);
		}
		return cast_alternative(src, pol, parent, type_list<Us...>());
	}
	template<typename Any>
	static handle cast_alternative(Any*, return_value_policy, handle, type_list<>) {
		return {};
	}

	template<typename Any>
	static handle cast(Any&& src, return_value_policy pol, handle parent) {
		namespace py = pybind11;

		if(auto res = cast_alternative(&src, pol, parent, cast_seq_v))
			return res;
		throw py::value_error("Could not convert `any` value to Python");
	}
};

template<>
struct type_caster<std::any> : any_caster<
		std::int64_t, bool, double, std::string, blue_sky::timestamp, blue_sky::timespan, blue_sky::sp_obj
		// object // <-- enables any Python object passthrough
> {};

NAMESPACE_END(detail::pybind11)
