/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef PY_BS_CONVERTER_EPS7T5L1
#define PY_BS_CONVERTER_EPS7T5L1

#include <boost/python/to_python_converter.hpp>
#include <boost/python/converter/implicit.hpp>
#include <boost/python/converter/registry.hpp>

namespace blue_sky { namespace python {

// register conversion from Python object to type described by conv_traits
// based on code by Roman Yakovenko
// http://www.language-binding.net/pyplusplus/troubleshooting_guide/automatic_conversion/automatic_conversion.html/
// indirect construction is used when type is created from another C++ type
// which is extracted from|exported to Python directly
template< class conv_traits >
class bspy_converter {
public:
	typedef typename conv_traits::type target_type;

	static void construct(PyObject* py_obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
		construct_(py_obj, data, conv_traits::create_type);
	}

	template< class result_t >
	static void construct_indirect(PyObject* py_obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
		construct_(py_obj, data, conv_traits::template create_type_indirect< result_t >);
	}

	static void* is_convertible(PyObject* py_obj) {
		if(conv_traits::is_convertible(py_obj))
			return py_obj;
		else
			return 0;
	}

	static void register_from_py() {
		namespace bp = boost::python;
		bp::converter::registry::push_back(
			  &is_convertible
			, &construct
			, bp::type_id< target_type >()
			);
	}

	template< class original_t >
	static void register_from_py_indirect() {
		using namespace boost::python;
		converter::registry::push_back(
			  &is_convertible
			, &construct_indirect< original_t >
			, type_id< target_type >()
			);
	}

	//---- to python converters (if type wasn't exported already)
	struct to_python
	{
		static PyObject* convert(target_type const &v) {
			return conv_traits::to_python(v);
		}
	};

	template< class original_t >
	struct to_python_indirect
	{
		static PyObject* convert(original_t const &v) {
			return conv_traits::template to_python_indirect< original_t >(v);
		}
	};

	static void register_to_py() {
		boost::python::to_python_converter< target_type, to_python >();
	}

	template< class original_t >
	static void register_to_py_indirect() {
		boost::python::to_python_converter< target_type, to_python_indirect< original_t > >();
	}

private:
	template< class create_f >
	static void construct_(PyObject* py_obj, boost::python::converter::rvalue_from_python_stage1_data* data, create_f create_type) {
		using namespace boost::python;

		typedef converter::rvalue_from_python_storage< typename conv_traits::type > storage_t;
		storage_t* the_storage = reinterpret_cast< storage_t* >(data);
		void* memory_chunk = the_storage->storage.bytes;

		// convert PyObject -> boost::python::object
		object bpy_obj(handle<>(borrowed(py_obj)));

		// create object using placement new
		create_type(memory_chunk, bpy_obj);
		data->convertible = memory_chunk;
	}
};

}}

#endif /* end of include guard: PY_BS_CONVERTER_EPS7T5L1 */
