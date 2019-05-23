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
NAMESPACE_BEGIN()

template<typename Propbook>
auto bind_propbook(py::module& m, const char* cl_name) {
	auto from_pydict = [](Propbook& tgt, py::dict src) {
		using Key = typename Propbook::key_type;
		using Value = typename Propbook::mapped_type;

		auto key_caster = py::detail::make_caster<Key>();
		auto value_caster = py::detail::make_caster<Value>();
		for(const auto& [sk, sv] : src) {
			if(!key_caster.load(sk, true) || !value_caster.load(sv, true)) continue;
			tgt[py::detail::cast_op<Key>(key_caster)] = py::detail::cast_op<Value>(value_caster);
		}
	};

	auto res = py::bind_map<Propbook>(m, cl_name, py::module_local(false))
		.def(py::init([&from_pydict](py::dict D) {
			auto B = std::make_unique<Propbook>();
			from_pydict(*B, std::move(D));
			return B;
		}))
		.def("to_dict", [](const Propbook& B) {
			using Key = typename Propbook::key_type;
			using Propmap = typename prop::propdict::underlying_type;
			return std::map<Key, Propmap>(B.begin(), B.end());
		})
		.def("from_dict", from_pydict)
	;

	py::implicitly_convertible<py::dict, Propbook>();
	return res;
}

NAMESPACE_END()

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

	// type_desccriptor bind
	py::class_< type_descriptor >(m, "type_descriptor")
		.def(py::init<>())
		.def(py::init<std::string_view>(), "type_name"_a)
		.def(py::init<
				std::string, const BS_TYPE_COPY_FUN&, const BS_GET_TD_FUN&, std::string
			>(), "type_name"_a, "copy_fun"_a, "parent_td_fun"_a, "description"_a = ""
		)
		.def_property_readonly_static(
			"nil", [](py::object){ return type_descriptor::nil(); }, py::return_value_policy::reference
		)
		.def_readonly("name", &type_descriptor::name)
		.def_readonly("description", &type_descriptor::description)
		.def("add_copy_constructor", (void (type_descriptor::*)(BS_TYPE_COPY_FUN) const)
			&type_descriptor::add_copy_constructor, "copy_fun"_a
		)
		.def("clone", [](const type_descriptor& td, bs_type_copy_param src){
			return (std::shared_ptr< objbase >)td.clone(src);
		})
		.def_property_readonly("is_nil", &type_descriptor::is_nil)
		.def_property_readonly("is_copyable", &type_descriptor::is_copyable)
		.def("parent_td", &type_descriptor::parent_td, py::return_value_policy::reference)
		.def(py::self == py::self)
		.def(py::self != py::self)
		.def(py::self < py::self)
		.def(py::self < std::string())
		.def(py::self == std::string())
		.def(py::self != std::string())
		.def("__repr__", [](const type_descriptor& td) { return td.name; })
	;
	// enable implicit conversion from string -> type_descriptor
	py::implicitly_convertible<std::string, type_descriptor>();

	// propdict binding
	py::bind_map<prop::propdict>(m, "propdict", py::module_local(false))
		.def(py::init<prop::propdict::underlying_type>())
		.def("has_key", &prop::propdict::has_key)
		.def("keys", &prop::propdict::keys)
		.def("to_dict", [](const prop::propdict& D) -> const prop::propdict::underlying_type& {
			return D;
		})
	;
	// allow passing compatible Python dict in place of `propdict` (and init propdict from that Py dict)
	py::implicitly_convertible<prop::propdict::underlying_type, prop::propdict>();
	// opaque bindings of propbooks
	bind_propbook<prop::propbook_s>(m, "propbook_s");
	bind_propbook<prop::propbook_i>(m, "propbook_i");
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

