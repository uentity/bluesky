/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "type_descriptor.h"
#include "py_bs_exports.h"

namespace blue_sky { namespace python {

void py_export_typed() {
	//class_< type_descriptor >("type_descriptor", no_init)
	//	.add_property("stype_", &type_descriptor::stype_)
	//	.add_property("short_descr_", &type_descriptor::short_descr_)
	//	.add_property("long_descr_", &type_descriptor::long_descr_)
	//	.def(self < std::string())
	//	.def(self < self)
	//	.def(self == self)
	//	.def(self == std::string())
	//	.def(self != self)
	//	.def(self != std::string());

	//class_< BS_TYPE_INFO >("bs_typeinfo");

	//class_< std::vector< BS_TYPE_INFO > >("vector_typeinfo")
	//	.def(vector_indexing_suite< std::vector< BS_TYPE_INFO > >());
}

}}
