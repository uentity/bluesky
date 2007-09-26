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

#include "py_bs_command.h"
#include "py_bs_exports.h"

namespace blue_sky {
namespace python {

//py_combase::py_combase() { //combase *tsp) : spcom(tsp) {
	 //printf ("%p\n%p\n",tsp,spcom.get());
//}

py_combase::py_combase(const sp_com &tsp) : spcom(tsp) {}

const sp_com &py_combase::get_spcom() {
	 return spcom;
}

py_combase py_combase::execute() {
	 printf ("%p\n",spcom.get());
	 return spcom.lock()->execute();
}

void py_combase::unexecute() {
	 spcom.lock()->unexecute();
}

bool py_combase::can_unexecute() const {
	 return spcom->can_unexecute();
}

void py_export_combase() {
	class_<py_combase>("combase", no_init)
		.def("execute",&py_combase::execute)
		.def("unexecute",&py_combase::unexecute)
		.def("can_unexecute",&py_combase::can_unexecute);

	class_<sp_com, noncopyable>("sp_com", no_init);
}

}	//namespace blue_sky::python
}	//namespace blue_sky
