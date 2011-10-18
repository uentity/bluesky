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

#ifndef PY_ARRAY_CONVERTER_QSXYZL0I
#define PY_ARRAY_CONVERTER_QSXYZL0I

#include "py_bs_converter.h"
#include "bs_kernel.h"

#include "bs_array.h"
#include "bs_nparray.h"
#include "bs_npvec.h"
#include "bs_npvec_shared.h"

namespace blue_sky { namespace python {

// implements converters for all types of arrays in BS
template< class T >
struct array_converters {
	typedef pyublas::numpy_array< T > backend_t;
	typedef bs_nparray< T > cont_t;
	typedef bs_array< T, bs_nparray > nparray_t;
	typedef smart_ptr< nparray_t > sp_nparray_t;

	/*-----------------------------------------------------------------
	 * bs_nparray <--> Python converter
	 *----------------------------------------------------------------*/
	template< template< class > class cont_traits = bs_nparray >
	struct nparray_traits {
		typedef bs_array< T, cont_traits > array_t;
		typedef bs_nparray< T > cont_t;
		// target type for bspy_converter
		typedef smart_ptr< array_t > type;

		static void create_type(void* memory_chunk, const boost::python::object& py_obj) {
			// create empty array and init it with Python array
			type sp_array = BS_KERNEL.create_object(array_t::bs_type());
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
			if(!v)
				return cont_t().to_python();
			//return v->handle().get();
			return v->to_python();
		}
	};

	// safe implicit conversion that copies data buffers
	template< template< class > class cont_traits >
	struct copy_traits {
		typedef bs_array< T, cont_traits > array_t;
		// target type for bspy_converter
		typedef smart_ptr< array_t > type;

		static void create_type(void* memory_chunk, const boost::python::object& py_obj) {
			// create empty array and init it with Python array
			cont_t src(py_obj.ptr());

			// copy data to destination array
			type dst = BS_KERNEL.create_object(array_t::bs_type());
			dst.lock()->resize(src.size());
			std::copy(src.begin(), src.end(), dst.lock()->begin());
			new(memory_chunk) type(dst);
		}

		static PyObject* to_python(type const& v) {
			if(!v)
				return cont_t().to_python();
			// create empty array and init it with copied data from v
			cont_t proxy(v->size());
			std::copy(v->begin(), v->end(), proxy.begin());
			return proxy.to_python();
		}

		static bool is_convertible(PyObject* py_obj) {
			return array_converters< T >::nparray_traits<>::is_convertible(py_obj);
		}
	};

	// safe implicit conversion that copies data buffers
	// preserve shape data
	template< template< class > class cont_traits >
	struct copy_traits_wshape {
		typedef bs_array< T, cont_traits > array_t;
		typedef cont_traits< T > traits_t;
		// target type for bspy_converter
		typedef smart_ptr< array_t > type;

		static void create_type(void* memory_chunk, const boost::python::object& py_obj) {
			// create empty array and init it with Python array
			cont_t src(py_obj.ptr());

			// copy data to destination array
			// along with shape info
			type dst = BS_KERNEL.create_object(array_t::bs_type());
			dst.lock()->init(traits_t(src.ndim(), src.dims(), src.data()));
			new(memory_chunk) type(dst);
		}

		static PyObject* to_python(type const& v) {
			if(!v)
				return cont_t().to_python();
			// create empty array and init it with copied data from v
			cont_t proxy(backend_t(v->ndim(), v->dims()));
			std::copy(v->begin(), v->end(), proxy.begin());
			return proxy.to_python();
		}

		static bool is_convertible(PyObject* py_obj) {
			return array_converters< T >::nparray_traits<>::is_convertible(py_obj);
		}
	};

	// conversions that reference data buffer
	template< template< class > class cont_traits >
	struct shared_traits {
		typedef bs_array< T, cont_traits > array_t;
		// target type for bspy_converter
		typedef smart_ptr< array_t > type;

		static void create_type(void* memory_chunk, const boost::python::object& py_obj) {
			// create proxy nparray and init it with Python array
			sp_nparray_t src = BS_KERNEL.create_object(nparray_t::bs_type());
			src.lock()->init(cont_t(py_obj.ptr()));
			// make empty destination array
			type dst = BS_KERNEL.create_object(array_t::bs_type());
			// set it's container to newly created proxy
			dst->init_inplace(src);
			new(memory_chunk) type(dst);
		}

		static PyObject* to_python(type const& v) {
			if(!v)
				return cont_t().to_python();
			// convert opaque ptr to data into Python object
			// then explicitly increment refcounter of sp_data
			// and pass ptr to refcounter as *desc argument of
			// desctructor function that will explicitly decrement refcounter
			const bs_refcounter* data_rc = static_cast< const bs_refcounter* >(v->get_container().get());
			data_rc->add_ref();
			PyObject* opaque_ptr = PyCObject_FromVoidPtr((void*)data_rc,
				array_converters::on_numpy_array_death);

			// build numpy array around raw data
			cont_t proxy(v->begin(), v->size());
			// set array's BASE to opaque_ptr to correctly free resourses
			PyArray_BASE(proxy.handle().get()) = opaque_ptr;

			// return created numpy array to Python
			return proxy.to_python();
		}

		static bool is_convertible(PyObject* py_obj) {
			return nparray_traits<>::is_convertible(py_obj);
		}
	};

	// conversions that return data from C++ -> Python by reference
	// and copy data when it comes from Python -> C++
	// ensure that data is alive when Python interpreter dies
	template< template< class > class cont_traits >
	struct semi_shared_traits_wshape {
		typedef bs_array< T, cont_traits > array_t;
		typedef cont_traits< T > traits_t;
		// target type for bspy_converter
		typedef smart_ptr< array_t > type;

		static void create_type(void* memory_chunk, const boost::python::object& py_obj) {
			// create empty array and init it with Python array
			cont_t src(py_obj.ptr());

			// copy data to destination array
			// along with shape info
			type dst = BS_KERNEL.create_object(array_t::bs_type());
			dst.lock()->init(traits_t(src.ndim(), src.dims(), src.data()));
			new(memory_chunk) type(dst);
		}

		static PyObject* to_python(type const& v) {
			// sanity check
			if(!v)
				return cont_t().to_python();
			// convert opaque ptr to data into Python object
			// then explicitly increment refcounter of sp_data
			// and pass ptr to refcounter as *desc argument of
			// desctructor function that will explicitly decrement refcounter
			const bs_refcounter* data_rc = static_cast< const bs_refcounter* >(v->get_container().get());
			data_rc->add_ref();
			PyObject* opaque_ptr = PyCObject_FromVoidPtr((void*)data_rc,
				array_converters::on_numpy_array_death);

			// make numpy array that references raw data
			backend_t proxy(v->ndim(), v->dims(), v->data());
			// set array's BASE to opaque_ptr to correctly free resourses
			PyArray_BASE(proxy.handle().get()) = opaque_ptr;

			// return created numpy array to Python
			return proxy.to_python().release();
		}

		static bool is_convertible(PyObject* py_obj) {
			return nparray_traits<>::is_convertible(py_obj);
		}
	};

	static void on_numpy_array_death(void* p_refcnt) {
		const bs_refcounter* data_rc = static_cast< const bs_refcounter* >(p_refcnt);
		data_rc->del_ref();
	}


	template< class conv_traits >
	static void make_helper() {
		namespace bp = boost::python;

		typedef typename conv_traits::array_t array_t;
		typedef bspy_converter< conv_traits > converter_t;

		// register actual conversions
		converter_t::register_from_py();
		converter_t::register_to_py();
		// register smart_ptr conversions to bases
		bp::implicitly_convertible< typename conv_traits::type, smart_ptr< typename array_t::arrbase > >();
		bp::implicitly_convertible< typename conv_traits::type, smart_ptr< objbase > >();
	}

	template< template< class > class cont_traits >
	static void make() {
		make_helper< typename deduce_conv_traits< cont_traits >::type >();
	}

	// register all converters
	static void make_known() {
		// register native conversions to/from bs_array< T, bs_nparray > first
		make_helper< nparray_traits<> >();

		// ref converter for bs_array_shared
		make_helper< shared_traits< bs_array_shared > >();

		// copy converter for bs_vector_shared
		make_helper< copy_traits< bs_vector_shared > >();
		// copy converter for vector_traits
		make_helper< copy_traits< vector_traits > >();
		// copy with shape for bs_npvec
		make_helper< copy_traits_wshape< bs_npvec > >();
		// semi-shared traits fo bs_npvec_shared
		make_helper< semi_shared_traits_wshape< bs_npvec_shared > >();
	}

private:
	template< template< class > class cont_traits >
	struct deduce_conv_traits {
		typedef cont_traits< T > traits_t;
		typedef typename traits_t::bs_array_base abase_t;

		enum { use_nparray = conversion< abase_t, bs_nparray< T > >::exists };
		enum { use_copy = conversion< abase_t, bs_vecbase< T > >::exists && !use_nparray };
		enum { use_shared = conversion< abase_t, bs_arrbase< T > >::exists && !use_nparray && !use_copy };
		// 0 - error (unknown traits), 1 - nparray, 2 - copy, 3 - shared
		enum { traits_code = use_nparray + use_copy*2 + use_shared*3 };

		// unknown traits - error
		template< int t, class = void >
		struct code2conv_traits {
			struct unknown_array_traits;
			enum { er = sizeof(unknown_array_traits) };
		};

		template< class unused >
		struct code2conv_traits< 1, unused > {
			typedef nparray_traits< cont_traits > type;
		};

		template< class unused >
		struct code2conv_traits< 2, unused > {
			typedef copy_traits< cont_traits > type;
		};

		template< class unused >
		struct code2conv_traits< 3, unused > {
			typedef shared_traits< cont_traits > type;
		};

		typedef typename code2conv_traits< traits_code >::type type;
	};
};

}} 	// eof blue_sky::python

#endif /* end of include guard: PY_ARRAY_CONVERTER_QSXYZL0I */

