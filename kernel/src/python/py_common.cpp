/// @file
/// @author uentity
/// @date 24.11.2016
/// @brief Python bindings for some common kernel API
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>
#include <bs/propdict.h>
#include <bs/python/property.h>
#include <bs/python/map.h>
#include "../kernel/python_subsyst_impl.h"

#include <ostream>
#include <iostream>
#include <sstream>

NAMESPACE_BEGIN(blue_sky)
using namespace std;

// def(str(self)) impl for plugin_descriptor
BS_HIDDEN_API ostream& operator<<(ostream& os, const plugin_descriptor& pd) {
	os << "{PLUGIN: " << pd.name << "; VERSIOM " << pd.version;
	os << "; INFO: " << pd.description;
	os << "; NAMESPACE: " << pd.py_namespace << '}';
	return os;
}

// def(str(self)) impl for type_descriptor
BS_HIDDEN_API ostream& operator<<(ostream& os, const type_descriptor& td) {
	if(td.is_nil())
		os << "BlueSky Nil type" << endl;
	else {
		os << "{TYPENAME: " << td.name;
		os << "; INFO: " << td.description << '}';
	}
	return os;
}

// def(str(self)) impl for type_info
BS_HIDDEN_API ostream& operator<<(ostream& os, const bs_type_info& ti) {
	os << "BlueSky C++ type_info: '" << ti.name() << "' at " << endl;
	return os;
}

NAMESPACE_BEGIN(python)

// dumb function for testing type_d-tor <-> Py list
typedef std::vector< type_descriptor > type_v;
type_v test_type_v(const type_v& v) {
	using namespace std;
	cout << "Size of type_descriptor list = " << v.size() << endl;
	for(ulong i = 0; i < v.size(); ++i)
		cout << v[i].name << ' ';
	cout << endl;
	return v;
}

auto pyinfinte() -> py::object {
	static auto value = [] {
		// return `datetime.timedelta.max` value
		auto dtm = py::module::import("datetime");
		auto py_dt = dtm.attr("timedelta");
		return py_dt.attr("max");
	}();

	return value;
}

// exporting function
void py_bind_common(py::module& m) {
	m.def("nil_type_info", &nil_type_info);
	m.def("is_nil", &is_nil);

	// bs_type_info binding
	// empty constructor creates nil type
	py::class_< bs_type_info >(m, "type_info")
		.def(py::init( []() { return std::make_unique<bs_type_info>(nil_type_info()); } ))
		.def_property_readonly("name", &bs_type_info::name)
		.def_property_readonly_static(
			"nil", [](py::object){ return nil_type_info(); }
		)
		.def(py::self == py::self)
		.def(py::self != py::self)
		.def(py::self < py::self)
		.def(py::self > py::self)
		.def(py::self <= py::self)
		.def(py::self >= py::self)
		.def("__repr__", &bs_type_info::name)
	;

	// plugin_descriptor binding
	py::class_< plugin_descriptor >(m, "plugin_descriptor")
		.def(py::init< const char* >(), "plugin_name"_a)
		.def(py::init< const std::string& >(), "plugin_name"_a)
		.def(py::init([](
			const bs_type_info& tag, const char* name, const char* version,
			const char* description, const char* py_namespace
		){
			return std::make_unique<plugin_descriptor>(tag, name, version, description, py_namespace);
		}), "tag_type_info"_a, "plug_name"_a, "version"_a, "description"_a = "", "py_namespace"_a = "")
		.def_readonly("name", &plugin_descriptor::name)
		.def_readonly("version", &plugin_descriptor::version)
		.def_readonly("description", &plugin_descriptor::description)
		.def_readonly("py_namespace", &plugin_descriptor::py_namespace)
		.def_property_readonly("is_nil", &plugin_descriptor::is_nil)
		.def_property_readonly_static(
			"nil", [](py::object){ return plugin_descriptor::nil(); }, py::return_value_policy::reference
		)
		.def(py::self < py::self)
		.def(py::self == py::self)
		.def(py::self != py::self)
		.def("__repr__", [](const plugin_descriptor& pd) { return pd.name; })
	;
	// enable implicit conversion from string -> plugin_descriptor
	py::implicitly_convertible<std::string, plugin_descriptor>();

	// [NOTE] important to bind *before* type_descriptor
	// propdict binding
	bind_rich_map<prop::propdict>(m, "propdict", py::module_local(false));
	// allow passing compatible Python dict in place of `propdict` (and init propdict from that Py dict)
	py::implicitly_convertible<prop::propdict::underlying_type, prop::propdict>();

	// opaque bindings of propbooks
	bind_rich_map<prop::propbook_s>(m, "propbook_s", py::module_local(false));
	bind_rich_map<prop::propbook_i>(m, "propbook_i", py::module_local(false));

	// type_desccriptor bind
	py::class_< type_descriptor >(m, "type_descriptor")
		.def(py::init<std::string_view>(), "type_name"_a)
		.def_property_readonly_static(
			"nil", [](py::object){ return type_descriptor::nil(); }, py::return_value_policy::reference
		)
		.def_readonly("name", &type_descriptor::name)
		.def_readonly("description", &type_descriptor::description)
		.def_property_readonly("is_nil", &type_descriptor::is_nil)
		.def_property_readonly("is_copyable", &type_descriptor::is_copyable)
	
		.def("parent_td", &type_descriptor::parent_td, py::return_value_policy::reference)
		.def("add_copy_constructor", (void (type_descriptor::*)(BS_TYPE_COPY_FUN) const)
			&type_descriptor::add_copy_constructor, "copy_fun"_a
		)

		.def("construct", [](const type_descriptor& self){
			return (std::shared_ptr< objbase >)self.construct();
		}, "Default construct type")

		.def("clone", [](const type_descriptor& td, bs_type_copy_param src){
			return (std::shared_ptr< objbase >)td.clone(src);
		})

		.def("assign", &type_descriptor::assign,
			"target"_a, "source"_a, "params"_a = prop::propdict{}, "Assign content from source to target")

		.def(py::self == py::self)
		.def(py::self != py::self)
		.def(py::self < py::self)
		.def(py::self < std::string())
		.def(py::self == std::string())
		.def(py::self != std::string())
		.def("__repr__", [](const type_descriptor& td) -> std::string {
			return "[" + td.name + "] [" + td.description + ']';
		})
	;
	// enable implicit conversion from string -> type_descriptor
	py::implicitly_convertible<std::string, type_descriptor>();

	m.def("isinstance", py::overload_cast<const sp_cobj&, const type_descriptor&>(&isinstance),
		"obj"_a, "td"_a, "Check if obj type match given type_descriptor");
	m.def("isinstance", py::overload_cast<const sp_cobj&, std::string_view>(&isinstance),
		"obj"_a, "obj_type_id"_a, "Check if obj is of given type name");

	// add marker for infinite timespan
	m.attr("infinite") = pyinfinte();

	// async tag
	py::class_<launch_async_t>(m, "launch_async_t");
	m.attr("launch_async") = launch_async;

	// unsafe tag
	py::class_<unsafe_t>(m, "unsafe_t");
	m.attr("unsafe") = unsafe;
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

