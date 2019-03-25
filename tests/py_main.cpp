/// @file
/// @author uentity
/// @date 25.03.2019
/// @brief Entry point for BS Python tests
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

// forward declare tests
void test_nparray(py::module&);
void test_inheritance(py::module&);

PYBIND11_MODULE(py_bs_tests, m) {
	test_nparray(m);
	test_inheritance(m);
}

