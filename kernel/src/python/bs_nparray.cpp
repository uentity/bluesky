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

//BLUE_SKY_TYPE_STD_CREATE_T_DEF(bs_nparray, (class));
//BLUE_SKY_TYPE_STD_COPY_T_DEF(bs_nparray, (class));

// bs_array< T, bs_nparray > instantiations
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (int, bs_nparray));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (float, bs_nparray));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (double, bs_nparray));

//BLUE_SKY_TYPE_IMPL_T_SHORT(bs_nparray_i, objbase, "BS numpy int array");
//BLUE_SKY_TYPE_IMPL_T_SHORT(bs_nparray_f, objbase, "BS numpy float array");
//BLUE_SKY_TYPE_IMPL_T_SHORT(bs_nparray_d, objbase, "BS numpy double array");

kernel::types_enum register_nparray() {
	kernel::types_enum te;
	te.push_back(bs_nparray_i::bs_type());
	te.push_back(bs_nparray_f::bs_type());
	te.push_back(bs_nparray_d::bs_type());
	return te;
}

namespace python {

// bs_nparray test function
template< class inp_array_t, class ret_array_t >
smart_ptr< ret_array_t > test_nparray(smart_ptr< inp_array_t > a, smart_ptr< inp_array_t > b) {
	// a little test on nparray resize
	smart_ptr< bs_nparray_d > tmp = BS_KERNEL.create_object(bs_nparray_d::bs_type());
	tmp->resize(10);
	tmp->resize(11);

	ulong sz = std::min(a->size(), b->size());
	smart_ptr< ret_array_t > res = BS_KERNEL.create_object(ret_array_t::bs_type());
	*res = *a;
	res->assign(*a);
	//*res = *a->clone();
	res->resize(sz + 1);
	for(ulong i = 0; i < sz; ++i)
		(*res)[i] = (*a)[i] + (*b)[i];
	(*res)[res->size() - 1] = 12345;
	//res->insert(1.);
	return res;
}

namespace {
namespace bp = boost::python;

//template< class T >
//static smart_ptr< T > create_bs_type() {
//	typedef smart_ptr< T > sp_t;
//	// assign it with newly created object
//	sp_t sp_obj = BS_KERNEL.create_object(T::bs_type());
//	if(!sp_obj)
//		bs_throw_exception("Can't create an instance of " + T::bs_type().stype_);
//	return sp_obj;
//}
//
//template< class T >
//static boost::python::object py_create_bs_type(boost::python::object py_obj) {
//	typedef smart_ptr< T > sp_t;
//	// extract smart_ptr embedded into python object
//	sp_t& sp_obj = extract< sp_t& > (py_obj);
//	if(!sp_obj)
//		bs_throw_exception("Can't extract smart_ptr< " + T::bs_type().stype_ + " > from Python object");
//
//	// assign it with newly created object
//	sp_obj = create_bs_type< T >();
//	return py_obj;
//}

using namespace blue_sky;
using namespace std;

/*-----------------------------------------------------------------
 * wrapper class to handle Python -> C++ nparray reference conversion
 *----------------------------------------------------------------*/
template< class T >
struct nparray_shared : public bs_nparray< T > {
	typedef bs_nparray< T > nparray_t;
	typedef smart_ptr< nparray_t > sp_nparray_t;
	typedef typename nparray_t::numpy_array_t numpy_array_t;

	typedef bs_arrbase< T > arrbase_t;
	typedef typename arrbase_t::sp_arrbase sp_arrbase;

	typedef typename nparray_t::size_type size_type;
	//typedef py_arrbase_handle< T > handle_t;

	nparray_shared(const sp_arrbase& sp_data)
		//: data_(sp_data)
	{

		//// create arrbase handle to tie lifetime of sp_data and array creates
		//handle_t* h = PyObject_New(handle_t, handle_t::py_type());
		//if(h && (PyObject*)h != Py_None) {
		//	// store reference to passed container
		//	h->data_ = sp_data;
		//	PyArray_BASE(this->handle().get()) = (PyObject*)h;
		//}
	}

	static void on_array_delete(void* raw_data, void* p_refcnt) {
		const bs_refcounter* data_rc = static_cast< const bs_refcounter* >(p_refcnt);
		data_rc->del_ref();
	}

	void resize(size_type new_size) {
		// call pyublas::numpy_array implementation
		// results in creating new array with copied data
		// c++ array instance will point to brand new data
		// sync with Python array is lost here
		numpy_array_t::resize(new_size);
	}

	//sp_arrbase clone() const {
	//	return new nparray_shared(data_);
	//}

	//sp_arrbase_t data_;
};

/*-----------------------------------------------------------------
 * bs_nparray <--> Python converter
 *----------------------------------------------------------------*/
template< class T >
struct bspy_nparray_traits {
	typedef bs_array< T, bs_nparray > nparray_t;
	typedef bs_nparray< T > cont_t;
	typedef smart_ptr< nparray_t > type;

	static void create_type(void* memory_chunk, const bp::object& py_obj) {
		// create empty array and init it with Python array
		type sp_array = BS_KERNEL.create_object(nparray_t::bs_type());
		sp_array.lock()->init(cont_t(py_obj.ptr()));
		new(memory_chunk) type(sp_array);
	}

	static bool is_convertible(PyObject* py_obj) {
		if(PyArray_Check(py_obj) && pyublas::is_storage_compatible< T >(py_obj)) {
			return true;
		}
		return false;
	}

	static PyObject* to_python(type const& v) {
		//return v->handle().get();
		return v->to_python();
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
			sp_array.lock()->init(cont_t(py_obj.ptr()));
			// copy data to destination array
			type sp_resarray = BS_KERNEL.create_object(varray_t::bs_type());
			sp_resarray->resize(sp_array->size());
			std::copy(sp_array->begin(), sp_array->end(), sp_resarray.lock()->begin());
			new(memory_chunk) type(sp_resarray);
		}

		static PyObject* to_python(type const& v) {
			// create empty array and init it with copied data from v
			sp_nparray_t sp_array = BS_KERNEL.create_object(nparray_t::bs_type());
			sp_array.lock()->init(v->size());
			std::copy(v->begin(), v->end(), sp_array.lock()->begin());
			return sp_array->to_python();
		}

		static bool is_convertible(PyObject* py_obj) {
			return bspy_nparray_traits< T >::is_convertible(py_obj);
		}
	};

	// safe implicit conversions that copies data buffers
	template< class varray_t >
	struct indirect_ref_traits {
		typedef smart_ptr< nparray_t > sp_nparray_t;
		//typedef bs_array< T, vector_traits > varray_t;
		typedef smart_ptr< varray_t > type;

		// safe implicit conversions that reference passed data
		static void create_type(void* memory_chunk, const bp::object& py_obj) {
			// create proxy nparray and init it with Python array
			sp_nparray_t sp_array = BS_KERNEL.create_object(nparray_t::bs_type());
			sp_array.lock()->init(cont_t(py_obj.ptr()));
			// make empty destination array
			type sp_resarray = BS_KERNEL.create_object(varray_t::bs_type());
			// set it's container to newly created proxy
			sp_resarray->init_inplace(sp_array);
			new(memory_chunk) type(sp_resarray);
		}

		static PyObject* to_python(type const& v) {
			// convert opaque ptr to data into Python object
			// then explicitly increment refcounter of sp_data
			// and pass ptr to refcounter as *desc argument of
			// desctructor function that will explicitly decrement refcounter
			const bs_refcounter* data_rc = static_cast< const bs_refcounter* >(v->get_container().get());
			data_rc->add_ref();
			PyObject* opaque_ptr = PyCObject_FromVoidPtr((void*)data_rc, on_array_delete);

			// build numpy array around raw data
			sp_nparray_t proxy = BS_KERNEL.create_object(nparray_t::bs_type());
			proxy->init(cont_t(v->begin(), v->size()));

			// set array's BASE to opaque_ptr to correctly free resourses
			PyArray_BASE(proxy->handle().get()) = opaque_ptr;

			// switch v's container to proxy
			// old container is controlled by proxy
			// via opaque_ptr
			v.lock()->init_inplace(proxy);

			return proxy->to_python();
		}

		static void on_array_delete(void* p_refcnt) {
			const bs_refcounter* data_rc = static_cast< const bs_refcounter* >(p_refcnt);
			data_rc->del_ref();
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
		typedef bspy_converter< indirect_ref_traits< varray_t > > array_converter_t;

		// register main conversions
		// NOTE: main should go before copy conversions
		// 'cause otherwise indirect_copy_traits::create_type will be called
		// when input paramter of type bs_arrbase< T > is encountered
		converter_t::register_from_py();
		converter_t::register_to_py();
		// register smart_ptr conversions to bases
		//bp::implicitly_convertible< type, smart_ptr< typename nparray_t::bs_array_t > >();
		bp::implicitly_convertible< type, smart_ptr< typename nparray_t::arrbase > >();
		bp::implicitly_convertible< type, smart_ptr< objbase > >();

		// register implicit copy conversions
		array_converter_t::register_from_py();
		array_converter_t::register_to_py();
		// register smart_ptr conversions to bases
		bp::implicitly_convertible< sp_varray_t, smart_ptr< typename varray_t::arrbase > >();
		bp::implicitly_convertible< sp_varray_t, smart_ptr< objbase > >();
	}
};

}

void py_export_nparray() {
	// export converters
	bspy_nparray_traits< int >::register_converters();
	bspy_nparray_traits< float >::register_converters();
	bspy_nparray_traits< double >::register_converters();

	// export test functions
	def("test_nparray_i", &test_nparray< bs_nparray_i, bs_array< int > >);
	def("test_nparray_f", &test_nparray< bs_nparray_f, bs_array< float > >);
	def("test_nparray_d", &test_nparray< bs_nparray_d, bs_array< double > >);
	//def("test_nparray_d", &test_nparray< bs_array< double, vector_traits >, bs_array< double, vector_traits > >);
	//def("test_nparray_d", &test_nparray< bs_nparray< double >, bs_array< double, vector_traits > >);
	//def("test_nparray_d", &test_nparray< double, bs_arrbase >);
}

}}

