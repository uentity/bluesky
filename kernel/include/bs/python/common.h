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

#include <bs/error.h>
#include <bs/kernel/misc.h>
#include <bs/python/kernel.h>

#include <bs/python/expected.h>
#include <bs/python/object_ptr.h>

#include "timetypes.h"

typedef void (*bs_init_py_fn)(void*);

/*-----------------------------------------------------------------------------
 *  macro definitions
 *-----------------------------------------------------------------------------*/
/// init Python subsystem in BS plugin
#define BS_INIT_PY(mod_name)                                                              \
static void bs_init_py_subsystem_##mod_name(pybind11::module&);                           \
BS_C_API_PLUGIN void bs_init_py_subsystem(void* py_plugin_module) {                       \
    if(!py_plugin_module) return;                                                         \
    bs_init_py_subsystem_##mod_name(*static_cast< pybind11::module* >(py_plugin_module)); \
}                                                                                         \
PYBIND11_MODULE(mod_name, m) {                                                            \
    bs_init_py_subsystem_##mod_name(m);                                                   \
}                                                                                         \
void bs_init_py_subsystem_##mod_name(pybind11::module& m)

/// adds some required objbase API, for ex. bs_type(), to Python class interface
#define BSPY_EXPORT_DEF_(T_tup)                                \
.def_property_readonly_static("bs_type", [](pybind11::object){ \
    return BOOST_PP_TUPLE_ENUM(T_tup)::bs_type();              \
}, pybind11::return_value_policy::reference)
// for types with <= 1 template params
#define BSPY_EXPORT_DEF(T) \
BSPY_EXPORT_DEF_((T))
// for types with > 1 template params
#define BSPY_EXPORT_DEF_T(T, T_spec_tupe) \
BSPY_EXPORT_DEF_((T<T_spec_tup>))

// register new error construcctor that takes user-defined error values enum
#define BSPY_REGISTER_ERROR_ENUM(E) [](){                                             \
    if(auto kmod = (pybind11::module*)blue_sky::kernel::k_pymod()) {                  \
        auto err_ctor = pybind11::init<E>();                                          \
        auto err_class = (pybind11::class_<std::error_code>)kmod->attr("error_code"); \
        err_ctor.execute(err_class);                                                  \
        pybind11::implicitly_convertible<E, std::error_code>();                       \
        pybind11::implicitly_convertible<E, blue_sky::error>();                       \
    }                                                                                 \
}();

/// add extra types that can passthrogh via `std::any`
// [NOTE] define this BEFORE including <bs/python/any.h>
#define BSPY_ANY_CAST_EXTRA(...) \
namespace pybind11::detail { struct any_cast_extra { using type = type_list<__VA_ARGS__>; }; }

NAMESPACE_BEGIN(blue_sky::python)

/// inject namespace aliases
namespace py = pybind11;
using namespace pybind11::literals;

[[maybe_unused]] inline static const auto nogil = py::call_guard<py::gil_scoped_release>();

NAMESPACE_END(blue_sky::python)
