// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

#include "bs_kernel.h"
#include "bs_kernel_tools.h"
#include "bs_tree.h"
#include "py_bs_exports.h"
#include "py_smart_ptr.h"

#include <boost/python/overloads.hpp>
#include <iostream>

namespace {
using namespace blue_sky;
using namespace std;

struct kernel_handle {
	kernel_handle() : k_(&give_kernel::Instance()) {}
	kernel_handle(kernel*) : k_(&give_kernel::Instance()) {}

	kernel* k_;
};

kernel* get_pointer(const kernel_handle& kf) {
	return kf.k_;
}

struct py_kernel_tools {
	static void print_loaded_types() {
		cout << kernel_tools::print_loaded_types();
	}

	static void walk_tree() {
		cout << kernel_tools::walk_tree();
	}

	static void print_registered_instances() {
		cout << kernel_tools::print_registered_instances();
	}
};

}

namespace boost { namespace python {

template< >
struct pointee< kernel_handle > {
	typedef blue_sky::kernel type;
};

}}

namespace blue_sky { namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(create_object_overl, create_object, 1, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(load_plugin_overl, load_plugin, 2, 3);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(load_plugins_overl, LoadPlugins, 0, 1);

void py_bind_kernel() {
	// plugin_types overloads
	kernel::types_enum (kernel::*plugin_types1)(const plugin_descriptor&) const = &kernel::plugin_types;
	kernel::types_enum (kernel::*plugin_types2)(const std::string&) const = &kernel::plugin_types;
	// create_object overloads
	sp_obj (kernel::*create_object1)(const type_descriptor&, bool, bs_type_ctor_param) const = &kernel::create_object;
	sp_obj (kernel::*create_object2)(const std::string&, bool, bs_type_ctor_param) const = &kernel::create_object;

	{
		// main kernel binding
		scope k_scope = class_<
			kernel,
			kernel_handle,
			boost::noncopyable
			>
		("kernel_iface", no_init)
			.def("loaded_plugins", &kernel::loaded_plugins)
			.def("plugin_types", plugin_types1)
			.def("plugin_types", plugin_types2)
			.def("find_type", &kernel::find_type)
			.def("registeres_types", &kernel::registered_types)
			.def("register_type", &kernel::register_type)
			.def("create_type", create_object1, create_object_overl())
			.def("create_type", create_object2, create_object_overl())
			.def("create_object_copy", &kernel::create_object_copy)
			.def("register_instance", &kernel::register_instance)
			.def("free_instance", &kernel::free_instance)
			.def("tree_gc", &kernel::tree_gc)
			.def("objinst_cnt", &kernel::objinst_cnt)
			.def("create_storage", &kernel::create_storage)
			.def("close_storage", &kernel::close_storage)
			.def("load_plugin", &kernel::load_plugin, load_plugin_overl())
			.def("unload_lugin", &kernel::unload_plugin)
			.def("load_plugins", &kernel::LoadPlugins, load_plugins_overl())
			.def("unload_plugins", &kernel::UnloadPlugins)
			.def("pert_str_dt", &kernel::pert_str_dt)
			.def("pert_idx_dt", &kernel::pert_idx_dt)
			.def("get_last_error", &kernel::get_last_error)
			.def("add_task", &kernel::add_task)
			.def("is_tq_empty", &kernel::is_tq_empty)
			.def("wait_tq_empty", &kernel::wait_tq_empty)
			.add_property("tree_root_link", &kernel::bs_root)
			.def("reg_signal", &kernel::reg_signal)
			.def("rem_signal", &kernel::rem_signal)
			//.def("get_memory_manager", &kernel::get_memory_manager)
			//.def("get_log", &kernel::get_log)
			//.def("get_tlog", &kernel::get_tlog)
			.def("register_disconnector", &kernel::register_disconnector)
			.def("unregister_disconnector", &kernel::unregister_disconnector)
		;

		// quick access to root node
		k_scope.attr("tree_root") = BS_KERNEL.bs_root()->node();

		// kernel_tools binding
		class_< py_kernel_tools >("tools", no_init)
			.def("print_loaded_types", &py_kernel_tools::print_loaded_types)
			.def("walk_tree", &py_kernel_tools::walk_tree)
			.def("print_registered_instances", &py_kernel_tools::print_registered_instances)
			.staticmethod("print_loaded_types")
			.staticmethod("walk_tree")
			.staticmethod("print_registered_instances")
		;
	}

	// shortcat to access kernel
	// function form - like in C++
	def("give_kernel", &give_kernel::Instance, return_value_policy< reference_existing_object >());
	// like global attribute - no need to write '()'
	scope().attr("kernel") = kernel_handle();
}

}} 	// eof namespace blue_sky::python

