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

BLUE_SKY_TYPE_IMPL_T_SHORT(bs_nparray< int >, bs_nparray< int >::bs_array_t, "BS numpy int array");
BLUE_SKY_TYPE_IMPL_T_SHORT(bs_nparray< float >, bs_nparray< float >::bs_array_t, "BS numpy float array");
BLUE_SKY_TYPE_IMPL_T_SHORT(bs_nparray< double >, bs_nparray< double >::bs_array_t, "BS numpy double array");

kernel::types_enum register_nparray() {
	kernel::types_enum te;
	te.push_back(bs_nparray< int >::bs_type());
	te.push_back(bs_nparray< float >::bs_type());
	te.push_back(bs_nparray< double >::bs_type());
	return te;
}

namespace python {

// bs_nparray test function
template< class inp_array_t, class ret_array_t >
smart_ptr< ret_array_t > test_nparray(smart_ptr< inp_array_t > a, smart_ptr< inp_array_t > b) {
	ulong sz = std::min(a->size(), b->size());
	smart_ptr< ret_array_t > res = BS_KERNEL.create_object(ret_array_t::bs_type());
	*res = *a;
	res->assign(*a);
	res = a->clone();
	res->resize(sz);
	for(ulong i = 0; i < sz; ++i)
		(*res)[i] = (*a)[i] + (*b)[i];
	//res->insert(1.);
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
		return bp::handle< >(v->handle()).release();
	}

	// safe implicit conversions that copies data buffers
	template< class varray_t >
	struct indirect_copy_traits {
		typedef smart_ptr< nparray_t > sp_nparray_t;
		//typedef bs_array< T, vector_traits > varray_t;
		typedef smart_ptr< varray_t > type;

		// safe implicit conversions that copies data buffers
		static void create_type(void* memory_chunk, const bp::object& py_obj) {
			// create empty array and init it with Python array
			sp_nparray_t sp_array = BS_KERNEL.create_object(nparray_t::bs_type());
			sp_array.lock()->init(handle<>(bp::borrowed(py_obj.ptr())));
			// copy data to destination array
			type sp_resarray = BS_KERNEL.create_object(varray_t::bs_type());
			sp_resarray->resize(sp_array->size());
			std::copy(sp_array->begin(), sp_array->end(), sp_resarray.lock()->begin());
			new(memory_chunk) type(sp_resarray);
		}

		static PyObject* to_python(type const& v) {
			// original_t is a smart_ptr to array with different traits
			// create empty array and init it with copied data from v
			sp_nparray_t sp_array = BS_KERNEL.create_object(nparray_t::bs_type());
			sp_array.lock()->init(v->size());
			std::copy(v->begin(), v->end(), sp_array.lock()->begin());
			return bp::handle< >(sp_array->handle()).release();
		}
	
		static bool is_convertible(PyObject* py_obj) {
			return bspy_nparray_traits< T >::is_convertible(py_obj);
		}
	};

	// register all converters
	static void register_converters() {
		typedef bspy_nparray_traits< T > this_t;
		typedef typename nparray_t::arrbase arrbase_t;
		typedef bspy_converter< this_t > converter_t;

		typedef bs_array< T > varray_t;
		typedef smart_ptr< varray_t > sp_varray_t;
		typedef bspy_converter< indirect_copy_traits< varray_t > > copy_converter_t;

		// register main conversions
		// NOTE: main should go before copy conversions
		// 'cause otherwise indirect_copy_traits::create_type will be called
		// when input paramter of type bs_arrbase< T > is encountered
		converter_t::register_from_py();
		converter_t::register_to_py();
		// register smart_ptr conversions to bases
		bp::implicitly_convertible< type, smart_ptr< typename nparray_t::bs_array_t > >();
		bp::implicitly_convertible< type, smart_ptr< typename nparray_t::arrbase > >();
		bp::implicitly_convertible< type, smart_ptr< objbase > >();

		// register implicit copy conversions
		copy_converter_t::register_from_py();
		copy_converter_t::register_to_py();
		// register smart_ptr conversions to bases
		bp::implicitly_convertible< sp_varray_t, smart_ptr< typename varray_t::arrbase > >();
		bp::implicitly_convertible< sp_varray_t, smart_ptr< objbase > >();

	}
};

}

void py_export_nparray() {
	//class_< bs_nparray< int >, bases< bs_nparray< int >::bs_array_t >, smart_ptr< bs_nparray< int > >, boost::noncopyable >
	//class_< bs_nparray< int >, smart_ptr< bs_nparray< int > >, boost::noncopyable >
	//	("nparray_i", no_init)
	//	.def("__init__", make_constructor(create_bs_type< bs_nparray< int > >))
	//	.def("test", &bs_nparray< int >::test)
	//	;
	//register_smart_ptr< bs_nparray< int > >();
	//auto_reg_smart_ptr< bs_nparray< int > >();
	//auto_reg_smart_ptr< bs_nparray< int >, bs_nparray< int >::bs_array_t >();

	// instead of exporting bs_nparray export converters to/from python objects
	bspy_nparray_traits< int >::register_converters();
	bspy_nparray_traits< float >::register_converters();
	bspy_nparray_traits< double >::register_converters();

	def("test_nparray_i", &test_nparray< bs_nparray< int >, bs_array< int > >);
	def("test_nparray_f", &test_nparray< bs_nparray< float >, bs_array< float > >);
	def("test_nparray_d", &test_nparray< bs_arrbase< double >, bs_array< double > >);
	//def("test_nparray_d", &test_nparray< bs_array< double, vector_traits >, bs_array< double, vector_traits > >);
	//def("test_nparray_d", &test_nparray< bs_nparray< double >, bs_array< double, vector_traits > >);
	//def("test_nparray_d", &test_nparray< double, bs_arrbase >);
}

}}

