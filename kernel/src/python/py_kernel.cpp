/// @file
/// @author uentity
/// @date 08.09.2017
/// @brief Python bindings for BS kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>
#include <bs/tree/node.h>
#include <bs/python/kernel.h>
#include <bs/python/any.h>
#include <boost/lexical_cast.hpp>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(python)

namespace {

type_tuple test_type_tuple(const type_tuple& t) {
	std::cout << t.type_name() << ' ' << t.plug_name() << std::endl;
	return t;
}

template<template<class> class array_traits, typename T>
void set_any_array_value(
	any_array<array_traits>& A, const typename any_array<array_traits>::key_type& k, T value
) {
	if(A.has_key(k))
		A.template ss<T>(k) = std::move(value);
	else
		A.insert_element(k, std::move(value));
}

template< template<class> class array_traits >
void bind_any_array(py::module& m, const char* type_name) {
	using any_array_t = any_array<array_traits>;
	using key_type = typename any_array_t::key_type;

	py::class_<any_array_t>(m, type_name)
		.def(py::init<>())
		.def("__bool__",
			[](const any_array_t& A) -> bool { return !A.empty(); },
			"Check whether the array is nonempty"
		)
		.def("__len__", (std::size_t (any_array_t::*)() const) &any_array_t::size)
		.def_property_readonly("size", (std::size_t (any_array_t::*)() const) &any_array_t::size)
		.def("__contains__", [](const any_array_t& A, const key_type& k) { return A.has_key(k); })
		.def("__getitem__", [](const any_array_t& A, const key_type& k) {
			using array_trait = typename any_array_t::trait;
			auto pval = array_trait::find(A, k);
			if(pval == A.end())
				throw py::key_error("There's no element with key = " + boost::lexical_cast<std::string>(k));

			return array_trait::iter2val(pval);
		})
		.def("__setitem__", [](any_array_t& A, const key_type& k, py::object value) {
			if(py::isinstance<py::bool_>(value))
				set_any_array_value(A, k, py::cast<bool>(value));
			else if(py::isinstance<py::int_>(value))
				set_any_array_value(A, k, py::cast<long>(value));
			else if(py::isinstance<py::float_>(value))
				set_any_array_value(A, k, py::cast<double>(value));
			else if(py::isinstance<py::str>(value))
				set_any_array_value(A, k, py::cast<std::string>(value));
			else {
				auto bsobj = py::cast<sp_obj>(value);
				if(bsobj) set_any_array_value(A, k, std::move(bsobj));
				else throw py::value_error("Only primitive types and BS objects can be stored in any_array");
			}
		})
	;
}

} // eof hodden namespace

void py_bind_kernel(py::module& m) {

	m.def("test_type_tuple", &test_type_tuple);

	bind_any_array<str_any_traits>(m, "str_any_array");
	bind_any_array<idx_any_traits>(m, "idx_any_array");

	// kernel
	py::class_<kernel, std::unique_ptr<kernel, py::nodelete>>(m, "kernel")
		.def("load_plugin", &kernel::load_plugin,
			"fname"_a, "init_py_subsyst"_a)
		.def("load_plugins", &kernel::load_plugins,
			"py_root_module"_a = nullptr)
		.def("unload_plugin", &kernel::unload_plugin,
			"plugin_descriptor"_a)
		.def("unload_plugins", &kernel::unload_plugins)
		.def("last_error", &kernel::last_error)
		.def("register_type",
			py::overload_cast<const type_descriptor&, const plugin_descriptor*>(&kernel::register_type),
			"type_descriptor"_a, "plugin_descriptor"_a = nullptr)
		.def("register_type",
			py::overload_cast<const type_descriptor&, const std::string&>(&kernel::register_type),
			"type_descriptor"_a, "plugin_name"_a)
		.def("find_type", &kernel::find_type, "type_name"_a)
		.def("loaded_plugins", &kernel::loaded_plugins)
		.def("registered_types", &kernel::registered_types)
		.def("plugin_types", py::overload_cast<const plugin_descriptor&>(&kernel::plugin_types, py::const_),
			"plugin_descriptor"_a)
		.def("plugin_types", py::overload_cast<const std::string&>(&kernel::plugin_types, py::const_),
			"plugin_name"_a)
		.def("register_instance", (int (kernel::*)(const sp_obj&)) &kernel::register_instance)
		.def("free_instance", (int (kernel::*)(const sp_obj&)) &kernel::free_instance)
		.def("instances", (kernel::instances_enum (kernel::*)(const BS_TYPE_INFO&) const) &kernel::instances)
		.def("pert_str_any_array", &kernel::pert_str_any_array, py::return_value_policy::reference_internal)
		.def("pert_idx_any_array", &kernel::pert_idx_any_array, py::return_value_policy::reference_internal)

		.def("clone_object", [](const kernel& k, bs_type_copy_param src) -> sp_obj {
			return k.clone_object(src);
		}, "src"_a, "Make object copy")

		.def("init", &kernel::init, "Call this manually in the very beginning of main()")
		.def("shutdown", &kernel::shutdown, "Call this manually before program ends (before exit from main)")
	;

	// expose kernel instance as attribute
	m.attr("kernel") = &BS_KERNEL;
	// and as function
	m.def("give_kernel", &give_kernel::Instance, py::return_value_policy::reference);

	// kernel tools module
	auto kt_mod = m.def_submodule("tools", "Kernel tools");
	kt_mod.def("print_loaded_types", &kernel_tools::print_loaded_types);
	kt_mod.def("print_link", [](const tree::sp_link& l) { kernel_tools::print_link(l, 0); });
	kt_mod.def("print_link", [](const tree::sp_node& n, std::string name = "/") {
		kernel_tools::print_link(std::make_shared<tree::hard_link>(name, n), 0);
	}, "node"_a, "root_name"_a = "/");
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

