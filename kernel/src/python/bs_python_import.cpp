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

#include "bs_common.h"
#include "bs_import_common.h"
#include "py_bs_exports.h"
#include "py_bs_assert.h"

#include "shared_vector.h"

#include <boost/python/enum.hpp>
#include <boost/python/exception_translator.hpp>

namespace blue_sky { namespace python {

// forward definition of exporting functions
//void py_export_common();
void py_export_typed();
void py_export_combase();
void py_export_kernel();
void py_export_abstract_storage();
void py_export_link();
void py_export_log();
void py_export_messaging();
void py_export_objbase();
void py_export_shell();
void py_export_tree();
void py_export_bs_array();

BLUE_SKY_INIT_PY_FUN
{
	register_exception_translator<bs_exception>(&exception_translator);

	class_< plugin_descriptor >("plugin_descriptor", no_init)
		.add_property("name_", &plugin_descriptor::name_)
		.add_property("version_", &plugin_descriptor::version_)
		.add_property("short_descr_", &plugin_descriptor::short_descr_)
		.add_property("long_descr_", &plugin_descriptor::long_descr_)
		.def(self < self)
		.def(self == self)
		.def(self != self);

	py_export_vectors ();

	class_< std::list<std::string> >("list_string")
		.def("__iter__", boost::python::iterator< std::list<std::string> >());

	python::py_export_error_codes ();
	python::py_export_assert ();

	//export the rest stuff
	py_export_messaging();
	py_export_objbase();
	py_export_typed();
	py_export_combase();
	py_export_kernel();
	py_export_abstract_storage();
	py_export_link();
	py_export_log();
	py_export_shell();
	py_export_tree();
	py_export_bs_array();
}

}}
