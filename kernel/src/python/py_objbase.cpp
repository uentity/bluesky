/// @file
/// @author uentity
/// @date 13.04.2017
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(python)

// hidden details
namespace {

sp_obj test_anyobj(const sp_obj& obj) {
	std::cout << obj->bs_resolve_type().name << std::endl;
	if(obj->bs_resolve_type().name == "objbase")
		std::cout << "got Python object: "
		<< py::str(std::static_pointer_cast<py_object<>>(obj)->pyobj.ptr()) << std::endl;
	return obj;
}

} // eof hidden namespace

void py_bind_objbase(py::module& m) {
	// inode binding
	py::class_<inode, std::unique_ptr<inode>>(m, "inode")
		.def(py::init<std::string, std::string>())
		.def_readonly("owner", &inode::owner)
		.def_readonly("group", &inode::group)
		.def_property_readonly("suid", [](const inode& i) { return i.suid; })
		.def_property_readonly("sgid", [](const inode& i) { return i.sgid; })
		.def_property_readonly("sticky", [](const inode& i) { return i.sticky; })
		.def_property_readonly("u", [](const inode& i) { return i.u; })
		.def_property_readonly("g", [](const inode& i) { return i.g; })
		.def_property_readonly("o", [](const inode& i) { return i.o; })
	;

	// explicit ctor defiition for multiple reuse
	const auto objbase_ctor1 = [](std::string custom_oid = "") -> sp_obj {
		return std::make_shared<py_object<>>(std::move(custom_oid));
	};
	const auto objbase_ctor2 = [](std::string custom_oid, const inode& i) -> sp_obj {
		return std::make_shared<py_object<>>(std::move(custom_oid), std::make_unique<inode>(i));
	};
	// objebase binding
	py::class_< objbase, py_object<>, sp_obj >(m, "objbase", py::multiple_inheritance())
		BSPY_EXPORT_DEF(objbase)
		// use init_alias to always construct trampoline class and have valid pyobj property
		.def(py::init(objbase_ctor1), "custom_oid"_a = "")
		.def(py::init(objbase_ctor2), "custom_oid"_a, "i"_a)
		// construct from any Python type
		.def(py::init_alias<py::object>())

		.def("bs_resolve_type", &objbase::bs_resolve_type, py::return_value_policy::reference)
		.def("bs_register_this", &objbase::bs_register_this)
		.def("bs_free_this", &objbase::bs_free_this)
		.def("swap", &objbase::swap)
		.def_property("pyobj", [](const objbase& src) -> py::object {
			if(src.bs_resolve_type() == objbase::bs_type()) {
				return static_cast<const py_object<>&>(src).pyobj;
			}
			return py::none();
		}, [](objbase& src, py::object value) {
			if(src.bs_resolve_type() == objbase::bs_type()) {
				static_cast<py_object<>&>(src).pyobj = std::move(value);
			}
		})
		.def("type_id", &objbase::type_id)
		.def("id", &objbase::id)
		.def_property_readonly("is_node", &objbase::is_node)
		.def_property_readonly("info", &objbase::info)
		// here we have to make a copy, no way to pass unique_ptr as param
		.def("set_info", [](objbase& obj, const inode& i) {
			obj.set_info(std::make_unique<inode>(i));
		}, "info"_a)
		// DEBUG
		.def_property_readonly("refs", [](objbase& src) { return src.shared_from_this().use_count() - 1; })
	;
	// add custom constructors for objbase that always constructs trampoline class
	auto& td = objbase::bs_type();
	td.add_constructor([]() -> sp_obj { return std::make_shared<py_object<>>(); });
	td.add_constructor(objbase_ctor1);
	td.add_constructor(objbase_ctor2);
	td.add_copy_constructor([](bs_type_copy_param src) -> sp_obj {
		return std::make_shared<py_object<>>(
			*static_cast<const py_object<>*>(src.get())
		);
	});

	// any Python instance can be converted to objbase
	py::implicitly_convertible< py::object, objbase >();

	m.def("test_anyobj", &test_anyobj);
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

