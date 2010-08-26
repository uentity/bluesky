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
