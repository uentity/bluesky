/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "bs_common.h"
#include "type_descriptor.h"
#include "bs_kernel.h"
#include "py_bs_exports.h"
#include "py_list_converter.h"
#include "py_pair_converter.h"
#include "py_smart_ptr.h"
#include "bs_misc.h"
#include "py_string_converter.h"

#include <boost/python/overloads.hpp>

#include <ostream>
#include <iostream>
#include <sstream>

namespace blue_sky {
using namespace std;

// def(str(self)) impl for plugin_descriptor
BS_HIDDEN_API ostream& operator<<(ostream& os, const plugin_descriptor& pd) {
	os << "{PLUGIN: " << pd.name_ << "; VERSIOM " << pd.version_;
	os << "; INFO: " << pd.short_descr_;
	if(pd.long_descr_.size() > 0)		os << "; LONG INFO: " << pd.long_descr_;
	os << "; NAMESPACE: " << pd.py_namespace_ << '}';
	return os;
}

// def(str(self)) impl for type_descriptor
BS_HIDDEN_API ostream& operator<<(ostream& os, const type_descriptor& td) {
	if(td.is_nil())
		os << "BlueSky Nil type" << endl;
	else {
		os << "{TYPENAME: " << td.stype_;
		os << "; INFO: " << td.short_descr_;
		if(td.long_descr_.size() > 0)
			os << "; LONG INFO: " << td.long_descr_ << '}';
	}
	return os;
}

// def(str(self)) impl for type_info
BS_HIDDEN_API ostream& operator<<(ostream& os, const bs_type_info& ti) {
	os << "BlueSky C++ layer type_info: '" << ti.name() << "' at " << &ti.get() << endl;
	return os;
}

// overloads for std::string <-> std::wstring converters
BOOST_PYTHON_FUNCTION_OVERLOADS(str2wstr_overl, str2wstr, 1, 2)
BOOST_PYTHON_FUNCTION_OVERLOADS(wstr2str_overl, wstr2str, 1, 2)

namespace python {
using namespace boost::python;

namespace {
struct type_tuple_traits : public pair_traits< type_tuple > {
	static PyObject* to_python(type const& v) {
		// Make bp::object
		bp::tuple py_tuple = boost::python::make_tuple(v.pd_, v.td_);
		// export it to Python
		return incref(py_tuple.ptr());
	}
};

class bs_refcounter_pyw : public bs_refcounter, public wrapper< bs_refcounter > {
public:
	void dispose() const {
		this->get_override("dispose");
	}

	//void add_ref() const {
	//	if(override f = this->get_override("add_ref"))
	//		f();
	//	else
	//		bs_refcounter::add_ref();
	//}
	//void def_add_ref() const {
	//	bs_refcounter::add_ref();
	//}

	//void del_ref() const {
	//	if(override f = this->get_override("del_ref"))
	//		f();
	//	else
	//		bs_refcounter::del_ref();
	//}
	//void def_del_ref() const {
	//	bs_refcounter::del_ref();
	//}
};

template< class T >
struct std_cont_converter {
	typedef bspy_converter< list_traits< std::vector< T > > > v_converter;
	typedef bspy_converter< list_traits< std::list< T >, 1 > > l_converter;
	typedef bspy_converter< list_traits< std::set< T >, 2 > > s_converter;

	static void go() {
		v_converter::register_to_py();
		l_converter::register_to_py();
		s_converter::register_to_py();
		// Python lists goes to C++ in most simple vector form
		v_converter::register_from_py();
	}
};

}

// dumb function for testing type_d-tor <-> Py list
typedef std::vector< type_descriptor > type_v;
type_v test_type_v(const type_v& v) {
	using namespace std;
	cout << "Size of type_descriptor list = " << v.size() << endl;
	for(ulong i = 0; i < v.size(); ++i)
		cout << v[i].stype_ << ' ';
	cout << endl;
	return v;
}

// exporting function
void py_bind_common() {
	// bs_type_info binding
	class_< bs_type_info >("type_info")
		.def("before", &bs_type_info::before)
		.def("name", &bs_type_info::name)
		.def("is_nil", &bs_type_info::is_nil)
		.def(self == self)
		.def(self != self)
		.def(self < self)
		.def(self > self)
		.def(self <= self)
		.def(self >= self)
		.def(str(self))
		;

	// register vector of type_info <-> Python list converters
	typedef bspy_converter< list_traits< std::vector< bs_type_info > > > tiv_converter;
	tiv_converter::register_from_py();
	tiv_converter::register_to_py();

	// plugin_descriptor binding
	class_< plugin_descriptor >("plugin_descriptor", no_init)
		.def_readonly("name", &plugin_descriptor::name_)
		.def_readonly("version", &plugin_descriptor::version_)
		.def_readonly("short_descr", &plugin_descriptor::short_descr_)
		.def_readonly("long_descr", &plugin_descriptor::long_descr_)
		.def_readonly("py_namespace", &plugin_descriptor::py_namespace_)
		.def(self < self)
		.def(self == self)
		.def(self != self)
		.def ("__repr__", &plugin_descriptor::get_name)
		.def(str(self))
		;

	// register vector of plugin descriptors <-> Python list converters
	typedef bspy_converter< list_traits< std::vector< plugin_descriptor > > > pdv_converter;
	pdv_converter::register_from_py();
	pdv_converter::register_to_py();

	// type_desccriptor bind
	class_< type_descriptor >("type_descriptor")
		.def(init< const std::string& >())
		.def_readonly("stype", &type_descriptor::stype_)
		.def_readonly("short_descr", &type_descriptor::short_descr_)
		.def_readonly("long_descr", &type_descriptor::long_descr_)
		.def("type", &type_descriptor::type)
		.def("is_nil", &type_descriptor::is_nil)
		.def("is_copyable", &type_descriptor::is_copyable)
		.def("name", &type_descriptor::name)
		.def("parent_td", &type_descriptor::parent_td)
		.def(self == self)
		.def(self != self)
		.def(self < self)
		.def(self < std::string())
		.def(self == std::string())
		.def(self != std::string())
		.def ("__repr__", &type_descriptor::name)
		.def(str(self))
		;

	// register vector of type descriptors <-> Python list converters
	typedef bspy_converter< list_traits< std::vector< type_descriptor > > > tdv_converter;
	tdv_converter::register_from_py();
	tdv_converter::register_to_py();

	// register pair< plugin_descriptor, type_descriptor > <-> Py tuple
	typedef bspy_converter< type_tuple_traits > type_tuple_converter;
	type_tuple_converter::register_from_py();
	type_tuple_converter::register_to_py();

	// register vector of type_tuples <-> Python list converters
	typedef bspy_converter< list_traits< std::vector< type_tuple >, 1 > > ttv_converter;
	ttv_converter::register_from_py();
	ttv_converter::register_to_py();

	//def("test_type_v", &test_type_v);

	class_< bs_refcounter_pyw, boost::noncopyable >("refcounter")
		//.def("add_ref", &bs_refcounter::add_ref, &bs_refcounter_pyw::def_add_ref)
		//.def("del_ref", &bs_refcounter::del_ref, &bs_refcounter_pyw::def_del_ref)
		.def("add_ref", &bs_refcounter::add_ref)
		.def("del_ref", &bs_refcounter::del_ref)
		.add_property("refs", &bs_refcounter::refs)
		.def("dispose", pure_virtual(&bs_refcounter::dispose))
	;

	// register converters for some C++ containers
	std_cont_converter< int >::go();
	std_cont_converter< unsigned int >::go();
	std_cont_converter< long >::go();
	std_cont_converter< long long >::go();
	std_cont_converter< unsigned long >::go();
	std_cont_converter< unsigned long long >::go();
	std_cont_converter< float >::go();
	std_cont_converter< double >::go();
	std_cont_converter< std::string >::go();
	std_cont_converter< std::wstring >::go();

	// register to/from UTF-8 converters
	def("str2wstr", &str2wstr, str2wstr_overl());
	def("wstr2str", &wstr2str, wstr2str_overl());

	// export converter between Python unicode string and std::string
	typedef bspy_converter< utf8_string_traits > utf8_converter;
	utf8_converter::register_from_py();
}

}}	// eof namespace blue_sky::python


