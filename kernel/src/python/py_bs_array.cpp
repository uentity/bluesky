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

#include "bs_array.h"
#include "py_bs_converter.h"
#include "bs_kernel.h"

namespace blue_sky { namespace python {

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
//---- converter traits for bs_nparray
using namespace blue_sky;

template< class T >
struct bspy_array_traits {
	typedef bs_array< T, numpy_array_traits > bs_array_t;
	typedef smart_ptr< bs_array_t > type;

	static void create_type(void* memory_chunk, const bp::object& py_obj) {
		// create empty array and init it with Python array
		type sp_array = BS_KERNEL.create_object(bs_array_t::bs_type());
		sp_array.lock()->init(bp::handle<>(bp::borrowed(py_obj.ptr())));
		new(memory_chunk) type(sp_array);
	}

	static PyObject* to_python(type const& v) {
		type sp_array = BS_KERNEL.create_object(bs_array_t::bs_type());
		*sp_array = *v;
		return bp::handle< >(sp_array->handle()).release();
	}

	static bool is_convertible(PyObject* py_obj) {
		if(!PyArray_Check(py_obj) || !pyublas::is_storage_compatible< T >(py_obj)) {
			return false;
		}
		return true;
	}

	// safe implicit conversions that copies data buffers
	template< class varray_t >
	struct indirect_copy_traits {
		typedef smart_ptr< bs_array_t > sp_array_t;
		//typedef bs_array< T, vector_traits > varray_t;
		typedef smart_ptr< varray_t > type;

		// safe implicit conversions that copies data buffers
		static void create_type(void* memory_chunk, const bp::object& py_obj) {
			// create empty array and init it with Python array
			sp_array_t sp_array = BS_KERNEL.create_object(bs_array_t::bs_type());
			sp_array.lock()->init(bp::handle<>(bp::borrowed(py_obj.ptr())));
			// copy data to destination array
			type sp_resarray = BS_KERNEL.create_object(varray_t::bs_type());
			sp_resarray->resize(sp_array->size());
			std::copy(sp_array->begin(), sp_array->end(), sp_resarray.lock()->begin());
			new(memory_chunk) type(sp_resarray);
		}

		static PyObject* to_python(type const& v) {
			// original_t is a smart_ptr to array with different traits
			// create empty array and init it with copied data from v
			sp_array_t sp_array = BS_KERNEL.create_object(bs_array_t::bs_type());
			sp_array.lock()->init(v->size());
			std::copy(v->begin(), v->end(), sp_array.lock()->begin());
			return bp::handle< >(sp_array->handle()).release();
		}
	
		static bool is_convertible(PyObject* py_obj) {
			return bspy_array_traits::is_convertible(py_obj);
		}
	};

	// register all converters
	static void register_converters() {
		typedef bspy_array_traits< T > this_t;
		typedef typename bs_array_t::arrbase arrbase_t;
		typedef bspy_converter< this_t > converter_t;

		// register main conversions
		// NOTE: main should go before copy conversions
		// 'cause otherwise indirect_copy_traits::create_type will be called
		// when input paramter of type bs_arrbase< T > is encountered
		converter_t::register_from_py();
		converter_t::register_to_py();
		// register smart_ptr conversions to bases
		bp::implicitly_convertible< type, smart_ptr< typename bs_array_t::arrbase > >();
		bp::implicitly_convertible< type, smart_ptr< objbase > >();

		// register implicit copy conversions for bs_array< T, vector_traits >
		typedef bs_array< T, vector_traits > varray_t;
		typedef smart_ptr< varray_t > sp_varray_t;
		typedef bspy_converter< indirect_copy_traits< varray_t > > copy_converter_t;

		copy_converter_t::register_from_py();
		copy_converter_t::register_to_py();
		// register smart_ptr conversions to bases
		bp::implicitly_convertible< sp_varray_t, smart_ptr< typename varray_t::arrbase > >();
		bp::implicitly_convertible< sp_varray_t, smart_ptr< objbase > >();
	}
};

}

void py_export_bs_array() {
	using namespace boost::python;
	// instead of exporting bs_nparray export converters to/from python objects
	bspy_array_traits< int >::register_converters();
	bspy_array_traits< float >::register_converters();
	bspy_array_traits< double >::register_converters();

	def("test_nparray_i", &test_nparray< bs_array< int >, bs_array< int > >);
	def("test_nparray_f", &test_nparray< bs_array< float >, bs_array< float > >);
	def("test_nparray_d", &test_nparray< bs_arrbase< double >, bs_array< double > >);
	//def("test_nparray_d", &test_nparray< bs_array< double, vector_traits >, bs_array< double, vector_traits > >);
	//def("test_nparray_d", &test_nparray< bs_nparray< double >, bs_array< double, vector_traits > >);
	//def("test_nparray_d", &test_nparray< double, bs_arrbase >);
}

}}


