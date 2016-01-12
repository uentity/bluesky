/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "py_bs_messaging.h"
#include "py_bs_object_base.h"
#include "bs_report.h"
#include "py_bs_exports.h"
#include "py_object_handler.h"

#include <boost/python/call_method.hpp>
#include <string>

#include "print_python_exception.h"

using namespace boost::python;
namespace bp = boost::python;

namespace blue_sky {
namespace python {

void python_slot::execute(const sp_mobj&, int sig, const sp_obj&) const {
	try {
		if (override f = this->get_override("execute"))
			f(sig);
		else
			//std::cerr << "Function execute does not exists, or is not correct!" << std::endl;
			BSERROR << "Function execute does not exists, or is not correct!" << bs_end;
	}
  catch (error_already_set const &) {
    BSERROR << "Error occured in 'execute' function, try to retrieve message..." << bs_end;
    detail::print_python_exception ();
  }
	catch (...) {
		//std::cerr << "There are some memory leaks!" << std::endl;
		BSERROR << "Unhandled exception occured in 'execute' function." << bs_end;
	}
}

py_bs_messaging::py_bs_messaging(const sp_mobj &msgng)
	: spmsg(msgng)
{}

py_bs_messaging::py_bs_messaging(const py_bs_messaging &msgng)
	: spmsg(msgng.spmsg)
{}

bool py_bs_messaging::subscribe(int signal_code, const python_slot& slot) const {
	//std::cout << "spslot is " << slot.spslot.get() << std::endl;
	bool result = spmsg->subscribe(signal_code,slot.spslot);//sp_slot(&slot));//slot.spslot);
  result &= spmsg->subscribe (objbase::on_delete, new tools::py_object_handler (bp::detail::wrapper_base_::get_owner (slot)));
	//BSOUT << "Subscribe: " << result << bs_end;
	return result;
}

bool py_bs_messaging::unsubscribe(int signal_code, const python_slot& slot) const {
	return spmsg->unsubscribe(signal_code,slot.spslot);//sp_slot(&slot));//slot.spslot);
}

ulong py_bs_messaging::num_slots(int signal_code) const {
	return spmsg->num_slots(signal_code);
}

bool py_bs_messaging::fire_signal(int signal_code, const py_objbase* param) const {
	BSOUT << "fire_signal call!" << bs_end;
	return spmsg->fire_signal(signal_code,param->sp);
}

std::vector< int > py_bs_messaging::get_signal_list() const {
	return spmsg->get_signal_list();
}

py_bs_messaging& py_bs_messaging::operator=(const sp_mobj& lhs) {
	spmsg = lhs;
	return *this;
}

void py_export_messaging() {
	class_<py_bs_messaging>("messaging",no_init)
		.def("subscribe",&py_bs_messaging::subscribe)
		.def("unsubscribe",&py_bs_messaging::unsubscribe)
		.def("num_slots",&py_bs_messaging::num_slots)
		.def("fire_signal",&py_bs_messaging::fire_signal)
		.def("get_signal_list",&py_bs_messaging::get_signal_list);

	class_<python_slot, noncopyable>("slot_wrap")
		.def("execute",pure_virtual(&bs_slot::execute));
}

}	//namespace blue_sky::python
}	//namespace blue_sky
