/// @file
/// @author uentity
/// @date 02.09.2016
/// @brief Test pybind11 module creation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <pybind11/pybind11.h>

int add(int i, int j) {
	return i + j;
}

namespace py = pybind11;

PYBIND11_PLUGIN(test_pymod) {
	py::module m("test_pymod", "pybind11 example plugin");
	auto subm = m.def_submodule("example");

	subm.def("add", &add, "A function which adds two numbers");

	return m.ptr();
}

