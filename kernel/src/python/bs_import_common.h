/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

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
