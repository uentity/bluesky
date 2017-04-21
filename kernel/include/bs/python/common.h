/// @file
/// @author uentity
/// @date 03.04.2017
/// @brief Common declarations for making Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <bs/objbase.h>
#include <bs/exception.h>
#include <bs/type_macro.h>

#include <pybind11/pybind11.h>
// for auto-support of std containers
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
// operator overloading support
#include <pybind11/operators.h>

typedef void (*bs_init_py_fn)(void*);

/*-----------------------------------------------------------------------------
 *  macro definitions
 *-----------------------------------------------------------------------------*/
// init Python subsystem in BS plugin
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

// this macro turns on `pyobj` property in any BS type T (`pyobj` is enabled for objbase)
// to use it you *have* to declare constructors with `py::init_alias`
// and *have trampoline class* derived from py_object<T>
#define BSPY_ENABLE_PYOBJ_(T_tup)                                                      \
.def_property("pyobj", [](const BOOST_PP_TUPLE_ENUM(T_tup)& src) -> py::object {       \
    return static_cast<const py_object< BOOST_PP_TUPLE_ENUM(T_tup) >&>(src).pyobj;     \
}, [](BOOST_PP_TUPLE_ENUM(T_tup)& src, py::object value) {                             \
    static_cast<py_object<BOOST_PP_TUPLE_ENUM(T_tup)>&>(src).pyobj = std::move(value); \
})
// for types with <= 1 template params
#define BSPY_ENABLE_PYOBJ(T) \
BSPY_ENABLE_PYOBJ_((T))
// for types with > 1 template params
#define BSPY_ENABLE_PYOBJ_T(T, T_spec_tup) \
BSPY_ENABLE_PYOBJ_((T<T_spec_tup>))

// adds some required objbase API, for ex. bs_type(), to Python class interface
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

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(python)

namespace py = pybind11;
using namespace pybind11::literals;

///*-----------------------------------------------------------------------------
// *  trampoline class for bindings for BS objects
// *-----------------------------------------------------------------------------*/
template<typename Object = objbase>
class BS_API py_object : public Object {
public:
	// objbase can carry any Python instance
	py::object pyobj = py::none();

	// import derived type ctors
	using Object::Object;
	py_object() = default;

	// construct from any Python object
	py_object(py::object o) : pyobj(std::move(o)) {}

	const type_descriptor& bs_resolve_type() const override {
		PYBIND11_OVERLOAD(
			const type_descriptor&,
			Object,
			bs_resolve_type
		);
	}
};

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

