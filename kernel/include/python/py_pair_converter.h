/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef PY_PAIR_CONVERTER_G71YCSIW
#define PY_PAIR_CONVERTER_G71YCSIW

#include "py_bs_converter.h"
#include <boost/python/tuple.hpp>

namespace blue_sky { namespace python {
namespace bp = boost::python;

template< class pair_t >
struct pair_traits {
	typedef pair_t type;

	typedef typename pair_t::first_type first_t;
	typedef typename pair_t::second_type second_t;

	static void create_type(void* mem_chunk, boost::python::object& obj) {
		// create c++ object
		type* res = new(mem_chunk) type(
				bp::extract< first_t >(obj[0]),
				bp::extract< second_t >(obj[1])
				);
		(void)res;
	}

	static bool is_convertible(PyObject* py_obj) {
		if( !PySequence_Check( py_obj ) ){
			return false;
		}

		if( !PyObject_HasAttrString( py_obj, "__len__" ) ){
			return false;
		}

		bp::object py_seq( bp::handle<>( bp::borrowed( py_obj ) ) );
		size_t sz = bp::len(py_seq);
		if(sz != 2) return false;
		if(!bp::extract< first_t >(py_seq[0]).check())
			return false;
		if(!bp::extract< second_t >(py_seq[1]).check())
			return false;
		return true;
	}

	static PyObject* to_python(type const& v) {
		// Make bp::object
		bp::tuple py_tuple = bp::make_tuple(v.first, v.second);
		// export it to Python
		return bp::incref(py_tuple.ptr());
	}
};

}} // eof blue_sky::python


#endif /* end of include guard: PY_PAIR_CONVERTER_G71YCSIW */

