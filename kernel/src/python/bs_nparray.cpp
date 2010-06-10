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

#include "bs_nparray.h"
#include "py_bs_exports.h"
#include "py_smart_ptr.h"
#include "py_bs_converter.h"
#include "bs_kernel.h"

//#include <iostream>

namespace blue_sky {

BLUE_SKY_TYPE_STD_CREATE_T_DEF(bs_nparray, (class));
BLUE_SKY_TYPE_STD_COPY_T_DEF(bs_nparray, (class));

// for base class
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (int, numpy_array_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (float, numpy_array_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (double, numpy_array_traits));

BLUE_SKY_TYPE_IMPL_T_SHORT(bs_nparray< int >, bs_nparray< int >::base_t, "BS numpy int array");
BLUE_SKY_TYPE_IMPL_T_SHORT(bs_nparray< float >, bs_nparray< float >::base_t, "BS numpy float array");
BLUE_SKY_TYPE_IMPL_T_SHORT(bs_nparray< double >, bs_nparray< double >::base_t, "BS numpy double array");

kernel::types_enum register_nparray() {
	kernel::types_enum te;
	te.push_back(bs_nparray< int >::bs_type());
	te.push_back(bs_nparray< float >::bs_type());
	te.push_back(bs_nparray< double >::bs_type());
	return te;
}

namespace python {

// bs_nparray test function
template< class T >
smart_ptr< bs_nparray< T > > test_nparray(smart_ptr< bs_nparray< T > > a, smart_ptr< bs_nparray< T > > b) {
	ulong sz = std::min(a->size(), b->size());
	smart_ptr< bs_nparray< T > > res = BS_KERNEL.create_object(bs_nparray< T >::bs_type());
	res->resize(sz);
	for(ulong i = 0; i < sz; ++i)
		(*res)[i] = (*a)[i] + (*b)[i];
	return res;
}

namespace {
namespace bp = boost::python;

template< class T >
static smart_ptr< T > create_bs_type() {
	typedef smart_ptr< T > sp_t;
	// assign it with newly created object
	sp_t sp_obj = BS_KERNEL.create_object(T::bs_type());
	if(!sp_obj)
		bs_throw_exception("Can't create an instance of " + T::bs_type().stype_);
	return sp_obj;
}

template< class T >
static boost::python::object py_create_bs_type(boost::python::object py_obj) {
	typedef smart_ptr< T > sp_t;
	// extract smart_ptr embedded into python object
	sp_t& sp_obj = extract< sp_t& > (py_obj);
	if(!sp_obj)
		bs_throw_exception("Can't extract smart_ptr< " + T::bs_type().stype_ + " > from Python object");

	// assign it with newly created object
	sp_obj = create_bs_type< T >();
	return py_obj;
}

//---- converter traits for bs_nparray
using namespace blue_sky;

template< class T >
struct bspy_nparray_traits {
	typedef bs_nparray< T > nparray_t;
	typedef smart_ptr< nparray_t > type;

	static void create_type(void* memory_chunk, const bp::object& py_obj) {
		// create empty array and init it with Python array
		type sp_array = BS_KERNEL.create_object(nparray_t::bs_type());
		sp_array.lock()->init(handle<>(bp::borrowed(py_obj.ptr())));
		new(memory_chunk) type(sp_array);
	}

	static bool is_convertible(PyObject* py_obj) {
		if(!PyArray_Check(py_obj) || !pyublas::is_storage_compatible< T >(py_obj)) {
			return false;
		}
		return true;
	}

	static PyObject* to_python(type const& v) {
		return bp::handle<>(v->handle()).release();
	}

	static void register_converters() {
		typedef bspy_nparray_traits< T > this_t;
		typedef bspy_converter< this_t > converter_t;

		converter_t::register_from_py();
		converter_t::register_to_py();
	}
};

}

void py_export_nparray() {
	//class_< bs_nparray< int >, bases< bs_nparray< int >::base_t >, smart_ptr< bs_nparray< int > >, boost::noncopyable >
	//class_< bs_nparray< int >, smart_ptr< bs_nparray< int > >, boost::noncopyable >
	//	("nparray_i", no_init)
	//	.def("__init__", make_constructor(create_bs_type< bs_nparray< int > >))
	//	.def("test", &bs_nparray< int >::test)
	//	;
	//register_smart_ptr< bs_nparray< int > >();
	//auto_reg_smart_ptr< bs_nparray< int > >();
	//auto_reg_smart_ptr< bs_nparray< int >, bs_nparray< int >::base_t >();

	// instead of exporting bs_nparray export converters to/from python objects
	bspy_nparray_traits< int >::register_converters();
	bspy_nparray_traits< float >::register_converters();
	bspy_nparray_traits< double >::register_converters();

	def("test_nparray_i", &test_nparray< int >);
	def("test_nparray_f", &test_nparray< float >);
	def("test_nparray_d", &test_nparray< double >);
}

}}

