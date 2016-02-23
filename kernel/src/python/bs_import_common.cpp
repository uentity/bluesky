/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

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
