/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

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
