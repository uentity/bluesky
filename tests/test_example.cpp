#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;

/*-----------------------------------------------------------------------------
 *  objbase
 *-----------------------------------------------------------------------------*/
class objbase : public std::enable_shared_from_this< objbase > {
	int prop_ = 42;
public:
	virtual std::string name() const {
		return "objbase";
	}

	int prop() const { return prop_; }
};
using sp_obj = std::shared_ptr<objbase>;
using sp_cobj = std::shared_ptr<const objbase>;

// trampoline
class py_prop {
	double pyprop_ = 42.84;
public:
	double pyprop() const { return pyprop_; }
};

template<class T = objbase>
class py_objbase : public T, public py_prop {
public:
	using T::T;

	std::string name() const override {
		PYBIND11_OVERLOAD(std::string, T, name);
	}
};

/*-----------------------------------------------------------------------------
 *  iface
 *-----------------------------------------------------------------------------*/
class iface {
public:
	int n = 42;
	virtual int f() const = 0;
};

// trampoline
template<class T = iface>
class py_iface : public T  {
public:
	using T::T;

	int f() const override {
		PYBIND11_OVERLOAD_PURE(int, iface, f);
	}
};

/*-----------------------------------------------------------------------------
 *  mytype
 *-----------------------------------------------------------------------------*/
class mytype : public iface, public objbase {
public:
	std::string name() const override {
		return "mytype";
	}
	int f() const override {
		return 42;
	}
};
using sp_my = std::shared_ptr<mytype>;
using sp_cmy = std::shared_ptr<const mytype>;

// trmapoline
class py_mytype : public py_iface<py_objbase<mytype>> {
public:
	using py_iface<py_objbase<mytype>>::py_iface;

	int f() const override {
		PYBIND11_OVERLOAD(int, mytype, f);
	}
};

/*-----------------------------------------------------------------------------
 *  mytype_simple
 *-----------------------------------------------------------------------------*/
class mytype_simple : public objbase {
public:
	long value;

	using objbase::objbase;
	mytype_simple(long value_) : value(value_) {}

	std::string name() const override {
		return "mytype_simple";
	}
};

// custom type cast
NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template<> struct type_caster<std::shared_ptr<mytype_simple>> {
	std::shared_ptr<mytype_simple> value;

	static PYBIND11_DESCR name() {
		return type_descr(_("mytype_simple"));
	}

	bool load(handle src, bool) {
		/* Extract PyObject from handle */
		PyObject *source = src.ptr();
		/* Try converting into a Python integer value */
		PyObject *tmp = PyNumber_Long(source);
		if (!tmp)
			return false;
		/* Now try to convert into a C++ int */
		value = std::make_shared<mytype_simple>(PyLong_AsLong(tmp));
		Py_DECREF(tmp);
		/* Ensure return code was OK (to avoid out-of-range errors etc) */
		return !PyErr_Occurred();
	}

	static handle cast(const std::shared_ptr<mytype_simple>& src, return_value_policy, handle) {
		return PyLong_FromLong(src->value);
	}

	operator std::shared_ptr<mytype_simple>() { return value; }
	template< typename > using cast_op_type = std::shared_ptr<mytype_simple>;
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)

/*-----------------------------------------------------------------------------
 *  functors
 *-----------------------------------------------------------------------------*/
class myslot1 {
public:
	using Param = sp_cmy;
	virtual void act(Param param) const = 0;
};

class myslot2 {
public:
	using Param = sp_cobj;
	virtual void act(Param param) const = 0;
};

// trmapoline
template<class Slot>
class py_myslot : public Slot {
public:
	using typename Slot::Param;
	using Slot::Slot;

	void act(Param obj) const override {
		// capture one reference for testing purposes
		//static sp_cobj tmp = obj;
		std::cout << "py_myslot.act: got obj at " << obj.get() << ", use count = " << obj.use_count() << std::endl;
		PYBIND11_OVERLOAD_PURE(void, Slot, act, std::move(obj));
	}
};

PYBIND11_PLUGIN(example) {
	py::module m("example");

	// bind objbase
	py::class_<objbase, py_objbase<>, std::shared_ptr<objbase>>(m, "objbase")
		.def(py::init_alias<>())
		.def("name", &objbase::name)
		.def_property_readonly("prop", &objbase::prop)
		.def_property_readonly("refs", [](const objbase& src) { return src.shared_from_this().use_count() - 1; })
		.def_property_readonly("pyprop", [](const objbase& src) {
			auto py_src = dynamic_cast<const py_prop*>(&src);
			return py_src->pyprop();
		})
	;

	// iface
	py::class_<iface, py_iface<>, std::shared_ptr<iface>>(m, "iface")
		.def("f", &iface::f)
	;

	// bind mytype
	py::class_<mytype, iface, objbase, py_mytype, sp_my>(m, "mytype")
		.def(py::init_alias<>())
		.def("name", &mytype::name)
		.def("f", &mytype::f)
		.def_property_readonly("n", [](mytype& src){ return src.n; })
		.def("test_slot1", [](mytype& src, std::shared_ptr<myslot1> s) {
			s->act(std::static_pointer_cast<mytype>(src.shared_from_this()));
		})
		.def("test_slot2", [](mytype& src, std::shared_ptr<myslot2> s) {
			s->act(src.shared_from_this());
		})
	;

	//py::class_<mytype_simple, objbase, py_objbase_<mytype_simple>, std::shared_ptr<mytype_simple> >(m, "mytype_simple")
	//	.def(py::init_alias<>())
	//	.def("name", &mytype_simple::name)
	//;
	m.def("test_objbase_pass", [](std::shared_ptr<objbase> obj) {
		std::cout << "Am I mytype_simple instance? " << std::boolalpha
			<< bool(std::dynamic_pointer_cast<mytype_simple>(obj)) << std::endl;
	});
	m.def("test_mytype_pass", [](std::shared_ptr<mytype_simple> obj) {
		std::cout << "Am I objbase instance? " << std::boolalpha
			<< bool(std::dynamic_pointer_cast<objbase>(obj)) << std::endl;
	});
	//py::implicitly_convertible< std::shared_ptr<mytype_simple>, objbase >();

	// bind slot
	py::class_<myslot1, py_myslot<myslot1>, std::shared_ptr<myslot1>>(m, "myslot1")
		.def(py::init_alias<>())
		.def("act", &myslot1::act)
	;
	py::class_<myslot2, py_myslot<myslot2>, std::shared_ptr<myslot2>>(m, "myslot2")
		.def(py::init_alias<>())
		.def("act", &myslot2::act)
	;

	return m.ptr();
}

