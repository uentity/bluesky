/// @file
/// @author uentity
/// @date 01.12.2019
/// @brief Transparent conversion of BS timespan <-> Python
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/chrono.h>
#include <bs/timetypes.h>

NAMESPACE_BEGIN(blue_sky::python)
namespace py = pybind11;

BS_API auto pyinfinte() -> py::object;

NAMESPACE_END(blue_sky::python)

NAMESPACE_BEGIN(pybind11::detail)

/// transforms C++ blue_sky::timespan <-> Python datetime.timedelta.max (bs.infinite)
template<> struct type_caster<blue_sky::timespan> : duration_caster<blue_sky::timespan> {
	using super = duration_caster<blue_sky::timespan>;
	using Type = blue_sky::timespan;

	bool load(handle src, bool convert) {
		if(src.is(blue_sky::python::pyinfinte())) {
			value = blue_sky::infinite;
			return true;
		}
		else return super::load(src, convert);
	}

	static handle cast(const Type& t, return_value_policy rvp, handle parent) {
		if(t == blue_sky::infinite) return blue_sky::python::pyinfinte().release();
		else return super::cast(t, rvp, parent);
	}
};

NAMESPACE_END(pybind11::detail)
