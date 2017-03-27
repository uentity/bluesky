/// @file
/// @author uentity
/// @date 02.09.2016
/// @brief Test pybind11 module creation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

namespace {

int add(int x1, int x2) {
	return x1 + x2;
}

int sub(int x1, int x2) {
	return x1 + x2;
}

PyObject* reenter_module() {
	//py::module m("test_pymod", "pybind11 example plugin");
	py::module m("test_pymod1");

	auto subm = m.def_submodule("example");

	subm.def("sub", &sub, "A function which subtracts two numbers");
	return m.ptr();
}

}

PYBIND11_PLUGIN(test_pymod) {
	py::module m("test_pymod", "pybind11 example plugin");
	auto subm = m.def_submodule("example");

	subm.def("add", &add, "A function which adds two numbers");

	auto testm = py::reinterpret_borrow< py::module >(reenter_module());
	m.attr("testm") = testm;

	return m.ptr();
}

