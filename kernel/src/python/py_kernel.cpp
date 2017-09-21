/// @file
/// @author uentity
/// @date 08.09.2017
/// @brief Python bindings for BS kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>
#include <bs/python/kernel.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(python)

namespace {

type_tuple test_type_tuple(const type_tuple& t) {
	std::cout << t.type_name() << ' ' << t.plug_name() << std::endl;
	return t;
}

}

void py_bind_kernel(py::module& m) {

	m.def("test_type_tuple", &test_type_tuple);

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
		.def("pert_str_any_array", &kernel::pert_str_any_array)
		.def("pert_idx_any_array", &kernel::pert_idx_any_array)
	;

	auto kt_mod = m.def_submodule("tools", "Kernel tools");
	kt_mod.def("print_loaded_types", &kernel_tools::print_loaded_types);

	// expose kernel instance as attribute
	m.attr("kernel") = &BS_KERNEL;
	// and as function
	m.def("give_kernel", &give_kernel::Instance, py::return_value_policy::reference);
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

