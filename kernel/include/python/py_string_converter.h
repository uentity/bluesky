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

