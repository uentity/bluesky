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

#ifndef _BS_IMPORT_COMMON_H
#define _BS_IMPORT_COMMON_H

#ifdef _MSC_VER
	#pragma warning(disable:4244)
	#pragma warning(disable:4267)
#endif

#include "bs_exception.h"
#include "type_descriptor.h"
#include <boost/python.hpp>

namespace blue_sky {
namespace python {

BS_API bool equal_types(const type_descriptor& , const type_descriptor& );

template<class T>
struct T_to_PyObject_ptr {
	static PyObject* convert(T const& src) {
		return boost::python::incref(boost::python::object(src).ptr());
	}
};

void exception_translator(blue_sky::bs_exception const&);
void std_exception_translator(std::exception const&);

}	//namespace blue_sky::python
}	//namespace blue_sky

#endif // _BS_IMPORT_COMMON_H
