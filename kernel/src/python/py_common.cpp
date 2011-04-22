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

#include "bs_common.h"
#include "type_descriptor.h"
#include "bs_kernel.h"
#include "py_bs_exports.h"
#include "py_list_converter.h"
#include "py_pair_converter.h"
#include "py_smart_ptr.h"
//#include <boost/python/python.h>

#include <ostream>
#include <iostream>

namespace blue_sky {
using namespace std;

// def(str(self)) impl for plugin_descriptor
ostream& operator<<(ostream& os, const plugin_descriptor& pd) {
	os << '{' << endl;
	os << "\tPlugin: " << pd.name_ << ", version " << pd.version_ << endl;
	os << '\t' << pd.short_descr_ << endl;
	if(pd.long_descr_.size() > 0)
		os << '\t' << pd.long_descr_ << endl;
	os << "\tNamespace: " << pd.py_namespace_ << endl;
	os << '}' << endl;
	return os;
}

// def(str(self)) impl for type_descriptor
ostream& operator<<(ostream& os, const type_descriptor& td) {
	if(td.is_nil())
		os << "\tBlueSky Nil type" << endl;
	else {
		os << '{' << endl;
		os << "\tBlueSky type: " << td.stype_ << endl;
		os << '\t' << td.short_descr_ << endl;
		if(td.long_descr_.size() > 0)
			os << '\t' << td.long_descr_ << endl;
		os << '}' << endl;
		// print type full chain
		if(!td.parent_td().is_nil()) {
			os << "Parent type_descriptor ->" << endl;
			os << td.parent_td();
		}
	}
	return os;
}

// def(str(self)) impl for type_info
ostream& operator<<(ostream& os, const bs_type_info& ti) {
	os << "BlueSky C++ layer type_info: '" << ti.name() << "' at " << &ti.get() << endl;
	return os;
}

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

	void add_ref() const {
		if(override f = this->get_override("add_ref"))
			f();
		else
			bs_refcounter::add_ref();
	}
	void def_add_ref() const {
		bs_refcounter::add_ref();
	}

	void del_ref() const {
		if(override f = this->get_override("del_ref"))
			f();
		else
			bs_refcounter::del_ref();
	}
	void def_del_ref() const {
		bs_refcounter::del_ref();
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
    .def ("__str__", &plugin_descriptor::get_name)
    .def ("__repr__", &plugin_descriptor::get_name)
		//.def(str(self))
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
		.def(str(self))
    .def ("__repr__", &type_descriptor::name)
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
		.def("add_ref", &bs_refcounter::add_ref, &bs_refcounter_pyw::def_add_ref)
		.def("del_ref", &bs_refcounter::del_ref, &bs_refcounter_pyw::def_del_ref)
		.add_property("refs", &bs_refcounter::refs)
		.def("dispose", pure_virtual(&bs_refcounter::dispose))
	;
}

}}	// eof namespace blue_sky::python


