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

// real trampoline class for objbase
// can hold any Python type and is implicitly convertible from py::object
class py_objbase : public py_object<objbase> {
private:
	py::object pyobj_ = py::none();

public:
	using py_object<objbase>::py_object;

	// construct from any Python object
	py_objbase(py::object o) : pyobj_(std::move(o)) {}

	// access stored Python instance
	py::object& pyobj() { return pyobj_; }
	const py::object& pyobj() const { return pyobj_; }
};

sp_obj test_anyobj(const sp_obj& obj) {
	std::cout << obj->bs_resolve_type().name << std::endl;
	if(obj->bs_resolve_type().name == "objbase")
		std::cout << "got Python object: "
		<< py::str(std::static_pointer_cast<py_objbase>(obj)->pyobj().ptr()) << std::endl;
	return obj;
}

} // eof hidden namespace

void py_bind_objbase(py::module& m) {
	// objebase binding
	py::class_< objbase, py_objbase, sp_obj >(m, "objbase")
		BSPY_EXPORT_DEF(objbase)
		// use init_alias to always construct trampoline class and have valid pyobj property
		.def(py::init_alias<>())
		.def(py::init_alias<py::object>())
		.def("bs_resolve_type", &objbase::bs_resolve_type)
		.def("bs_register_this", &objbase::bs_register_this)
		.def("bs_free_this", &objbase::bs_free_this)
		.def("swap", &objbase::swap)
		.def_property("pyobj", [](const objbase& src) -> py::object {
			if(src.bs_resolve_type() == objbase::bs_type()) {
				return static_cast<const py_objbase&>(src).pyobj();
			}
			return py::none();
		}, [](objbase& src, py::object value) {
			if(src.bs_resolve_type() == objbase::bs_type()) {
				static_cast<py_objbase&>(src).pyobj() = std::move(value);
			}
		})
	;

	py::implicitly_convertible< py::object, objbase >();

	m.def("test_anyobj", &test_anyobj);
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

