/// @file
/// @author uentity
/// @date 03.04.2017
/// @brief Common declarations for making Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <pybind11/pybind11.h>
// for auto-support of std containers
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
// operator overloading support
#include <pybind11/operators.h>

/*-----------------------------------------------------------------------------
 *  helper macro definitions
 *-----------------------------------------------------------------------------*/
#define BSPY_EXPORT_DEF(T)                     \
.def_property_readonly_static("bs_type", [](pybind11::object){ return T::bs_type(); })
//.def_static("bs_type", &T::bs_type)

#define BS_INIT_PY(mod_name)                                                        \
extern "C" const blue_sky::plugin_descriptor* bs_get_plugin_descriptor();           \
static void init_py_subsystem_impl(pybind11::module&);                              \
BS_C_API_PLUGIN void bs_init_py_subsystem(void* py_plugin_module) {                 \
	if(!py_plugin_module) return;                                                   \
	init_py_subsystem_impl(*static_cast< pybind11::module* >(py_plugin_module));    \
}                                                                                   \
PYBIND11_PLUGIN(mod_name) {                                                         \
	pybind11::module m(#mod_name, bs_get_plugin_descriptor()->description.c_str()); \
	bs_init_py_subsystem(&m);                                                       \
	return m.ptr();                                                                 \
}                                                                                   \
void init_py_subsystem_impl(pybind11::module& m)

typedef void (*bs_init_py_fn)(void*);


NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(python)

namespace py = pybind11;
using namespace pybind11::literals;

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

