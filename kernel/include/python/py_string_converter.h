/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef PY_STRING_CONVERTER_I0QJ2M62
#define PY_STRING_CONVERTER_I0QJ2M62

#include "py_bs_converter.h"
#include "bs_misc.h"

namespace blue_sky { namespace python {
namespace bp = boost::python;

struct utf8_string_traits {
	typedef std::string type;
	typedef std::wstring wtype;

	static void create_type(void* mem_chunk, boost::python::object& obj) {
		// create c++ string from Python unicode string
		// boost::python knows how to convert unicode string to std::wstring
		// so use 2-steps procedure:
		// 1. convert Python object -> std::wstring
		wtype u8s = bp::extract< wtype >(obj);
		// 2. convert wstring to string using BS converter
		// NOTE: using native system encoding, hence wstr2str_n
		new(mem_chunk) type(wstr2str(u8s));
	}

	static bool is_convertible(PyObject* py_obj) {
		// accept only unicode strings
		if(!PyUnicode_Check(py_obj) || !bp::extract< wtype >(py_obj).check())
			return false;
		return true;
	}

	static PyObject* to_python(type const& v) {
		// use embedded conversion from string -> Python
		return bp::incref(bp::object(v).ptr());
	}
};

}} // eof blue_sky::python

#endif /* end of include guard: PY_STRING_CONVERTER_I0QJ2M62 */

