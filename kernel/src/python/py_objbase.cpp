/// @file
/// @author uentity
/// @date 13.04.2017
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>
#include <bs/python/expected.h>
#include <bs/serialize/object_formatter.h>

#include <pybind11/functional.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(python)

void py_bind_objbase(py::module& m) {
	// explicit ctor defiition for multiple reuse
	const auto objbase_ctor1 = [](std::string custom_oid = "") -> sp_obj {
		return std::make_shared<py_object<>>(std::move(custom_oid));
	};
	// objebase binding
	py::class_< objbase, py_object<>, sp_obj >(m, "objbase", py::multiple_inheritance())
		BSPY_EXPORT_DEF(objbase)
		.def(py::init(objbase_ctor1), "custom_oid"_a = "")

		.def("bs_resolve_type", &objbase::bs_resolve_type, py::return_value_policy::reference)
		.def("bs_register_this", &objbase::bs_register_this)
		.def("bs_free_this", &objbase::bs_free_this)
		.def("swap", &objbase::swap)
		.def("type_id", &objbase::type_id)
		.def("id", &objbase::id)
		.def_property_readonly("is_node", &objbase::is_node)
		.def_property_readonly("info", &objbase::info)
		// DEBUG
		.def_property_readonly("refs", [](objbase& src) { return src.shared_from_this().use_count(); })
	;
	// add custom constructors for objbase that always constructs trampoline class
	auto& td = objbase::bs_type();
	td.add_constructor([]() -> sp_obj { return std::make_shared<py_object<>>(); });
	td.add_constructor(objbase_ctor1);
	td.add_copy_constructor([](bs_type_copy_param src) -> sp_obj {
		return std::make_shared<py_object<>>( *static_cast<const py_object<>*>(src.get()) );
	});

	// object formatters
	py::class_<object_formatter>(m, "object_formatter")
		.def(py::init([](
			std::string fmt_name, object_saver_fn saver, object_loader_fn loader, bool stores_node = false
		) {
			return object_formatter{ std::move(fmt_name), std::move(saver), std::move(loader), stores_node };
		}), "fmt_name"_a, "saver_fn"_a, "loader_fn"_a, "stores_node"_a = false)
		.def_readonly("name", &object_formatter::name,
			"Formatter name treated by default as file extension")
		.def_readonly("stores_node", &object_formatter::stores_node,
			"For node-derived objects: false (default) if object file doesn't include leafs, true if include")
		.def("save", &object_formatter::save, "obj"_a, "obj_fname"_a, "fmt_name"_a)
		.def("load", &object_formatter::load, "obj"_a, "obj_fname"_a, "fmt_name"_a)
	;

	m.def("install_formatter", &install_formatter, "obj_type"_a, "obj_formatter"_a);
	m.def("uninstall_formatter", &uninstall_formatter, "obj_type_id"_a, "fmt_name"_a);
	m.def("formatter_installed", &formatter_installed, "obj_type_id"_a, "fmt_name"_a);
	m.def("list_installed_formatters", &list_installed_formatters, "obj_type_id"_a);
	m.def("get_formatter", &get_formatter, "obj_type_id"_a, "fmt_name"_a,
		py::return_value_policy::reference);
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

