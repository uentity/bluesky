/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "bs_kernel.h"
#include "py_bs_exports.h"
#include "py_bs_converter.h"
#include "py_smart_ptr.h"

// DEBUG
#include <iostream>

namespace blue_sky { namespace python {
namespace bp = boost::python;
using namespace std;

/*******************************************************************
**	this class is designed to hold reference to any Python object
*******************************************************************/
class py_anyobject : public objbase {
public:
	explicit py_anyobject(PyObject* obj)
		: pyobj_(bp::borrowed< >(obj))
	{}

	explicit py_anyobject(const bp::object& obj)
		: pyobj_(bp::borrowed< >(obj.ptr()))
	{}

	bp::object get() const {
		//std::cout << "py_anyobj.get = " << pyobj_.get() << std::endl;
		return bp::object(pyobj_);
	}

	void set(PyObject* obj) {
		release();
		py_anyobject(obj).swap(*this);
	}

	PyObject* release() const {
		return pyobj_.release();
	}

	void swap(py_anyobject& lhs) {
		objbase::swap(lhs);
		std::swap(pyobj_, lhs.pyobj_);
	}

	// Life support for Python object
	mutable bp::handle< > pyobj_;

	BLUE_SKY_TYPE_DECL(py_anyobject)
};
// default ctor - unused
py_anyobject::py_anyobject(bs_type_ctor_param)
{}
// copy ctor
py_anyobject::py_anyobject(const py_anyobject& src)
	: bs_refcounter(), objbase(src)
{ *this = src; }

namespace {
/*******************************************************************
**	conversion traits
*******************************************************************/
struct sp_anyobject_conv_traits {
	typedef smart_ptr< py_anyobject > type;

	static void create_type(void* mem_chunk, bp::object& obj) {
		//sp_obj body = bp::extract< sp_obj >(obj);
		//if(body && body->bs_resolve_type().stype_ == "py_anyobject")
		//	new(mem_chunk) type(body);
		//else
			new(mem_chunk) type(new py_anyobject(obj));
	}

	// can hold any Py object
	static bool is_convertible(PyObject* py_obj) {
		return true;
	}

	static PyObject* to_python(type const& v) {
		return v->get().ptr();
	}
};

struct anyobject_conv_traits {
	typedef py_anyobject type;

	static void create_type(void* mem_chunk, bp::object& obj) {
		new(mem_chunk) type(obj);
	}

	// can hold any Py object
	static bool is_convertible(PyObject* py_obj) {
		return true;
	}

	static PyObject* to_python(type const& v) {
		return v.get().ptr();
	}
};

struct spobj_spec_traits {
	typedef sp_obj type;
	
	static PyObject* to_python(type const& v) {
		if(v->bs_resolve_type().stype_ == "py_anyobject")
			return smart_ptr< py_anyobject >(v, bs_static_cast())->pyobj_.get();
		else {
			return bp::converter::registry::lookup(bp::type_id< sp_obj >()).to_python(
				static_cast<void const*>(&v)
			);
		}
	}
};

sp_obj test_anyobj(const sp_obj& o) {
	cout << "test_anyobj!" << endl;
	return o;
}

PyObject* anyget(const smart_ptr< py_anyobject >& obj) {
	return obj->pyobj_.get();
}

} 	// eof hidden namespace

/*******************************************************************
**	implementation
*******************************************************************/
BLUE_SKY_TYPE_STD_CREATE(py_anyobject);
BLUE_SKY_TYPE_STD_COPY(py_anyobject);

BLUE_SKY_TYPE_IMPL(py_anyobject, objbase, "py_anyobject", "Holds a refernce to any Python object in BS tree", "");

void py_bind_anyobject() {
	// register py_anyobject class
	bp::class_< py_anyobject, bases< objbase >, boost::noncopyable >
		("pyobj", init< PyObject* >())
		.def("bs_type", &py_anyobject::bs_type)
		.staticmethod("bs_type")
		.add_property("get", &py_anyobject::get, &py_anyobject::set)
	;

	// register converters
	typedef bspy_converter< sp_anyobject_conv_traits > converter;
	converter::register_from_py();
	converter::register_to_py();
	// implicit conversion to sp_obj
	//bp::register_ptr_to_python< smart_ptr< py_anyobject > >();
	bp::implicitly_convertible< smart_ptr< py_anyobject >, sp_obj >();
	//bp::implicitly_convertible< sp_obj, smart_ptr< py_anyobject > >();

	//typedef bspy_converter< anyobject_conv_traits > converter1;
	//converter1::register_from_py();
	//converter1::register_to_py();

	// register special sp_obj converter
	typedef bspy_converter< spobj_spec_traits > spec_converter;
	//spec_converter::register_to_py();
	//bp::converter::registry::insert(&spobj_spec_traits::to_python, bp::type_id< sp_obj >());

	def("test_anyobj", &test_anyobj);
	def("anyget", &anyget);
}

}} /* blue_sky::python */

