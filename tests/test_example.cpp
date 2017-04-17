#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;

// classes
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

class iface {
public:
	int n = 42;
	virtual int f() const = 0;
};

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

// functors
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

// trampoline
template<class T = objbase>
class py_objbase : public T {
public:
	using T::T;

	std::string name() const override {
		PYBIND11_OVERLOAD(std::string, T, name);
	}
};

template<class T = objbase>
class py_objbase_ : public py_objbase<T> {
	double pyprop_ = 42.84;

public:
	using py_objbase<T>::py_objbase;

	double pyprop() const { return pyprop_; }
};

template<class T = iface>
class py_iface : public T  {
public:
	using T::T;

	int f() const override {
		PYBIND11_OVERLOAD_PURE(int, iface, f);
	}
};

// bind mytype
class py_mytype : public py_iface<py_objbase_<mytype>> {
public:
	using py_iface<py_objbase_<mytype>>::py_iface;

	int f() const override {
		PYBIND11_OVERLOAD(int, mytype, f);
	}
};

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
	py::class_<objbase, py_objbase_<>, std::shared_ptr<objbase>>(m, "objbase")
		.def(py::init_alias<>())
		.def("name", &objbase::name)
		.def_property_readonly("prop", &objbase::prop)
		.def_property_readonly("refs", [](const objbase& src) { return src.shared_from_this().use_count() - 1; })
		.def_property_readonly("pyprop", [](const py_objbase_<>& src) {
			return src.pyprop();
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

