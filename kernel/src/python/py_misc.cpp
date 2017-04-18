/// @file
/// @author uentity
/// @date 18.04.2017
/// @brief Misc kernel Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(python)

void py_bind_misc(py::module& m) {
	m.def("gettime", &gettime);
	m.def("system_message", &system_message);
	m.def("last_system_message", &last_system_message);

	m.def("wstr2str", &wstr2str, "text"_a, "loc_name"_a  = "utf-8");
	m.def("str2wstr", &str2wstr, "text"_a, "loc_name"_a  = "utf-8");

	m.def("ustr2str", &ustr2str, "text"_a, "loc_name"_a  = "utf-8");
	m.def("str2ustr", &str2ustr, "text"_a, "loc_name"_a  = "utf-8");

	m.def("str2str", &str2str, "text"_a, "out_loc_name"_a, "in_loc_name"_a = "");
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

