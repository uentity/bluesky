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

#include "bs_import_common.h"

namespace blue_sky {
namespace python {

BS_API bool equal_types(const type_descriptor& left, const type_descriptor& right) {
	return (left == right);
}

void exception_translator(blue_sky::bs_exception const& e) {
	PyErr_SetString(PyExc_RuntimeError, e.what());
}

void std_exception_translator(std::exception const& e) {
	PyErr_SetString(PyExc_RuntimeError, e.what());
}

}	//namespace blue_sky::python
}	//namespace blue_sky
