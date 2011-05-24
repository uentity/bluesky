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
#ifdef BSPY_EXPORTING_PLUGIN
#include <boost/python.hpp>
#endif

#include "bs_messaging.h"
#include "bs_object_base.h"
#include "py_bs_exports.h"
#include "py_smart_ptr.h"
//#include <boost/python/python.h>

// DEBUG
//#include <iostream>

namespace blue_sky { namespace python {

using namespace boost::python;

namespace {
using namespace std;
// bs_slot wrapper
class bs_slot_pyw : public bs_slot, public wrapper< bs_slot > {
public:
	void execute(const sp_mobj& sender, int signal_code, const sp_obj& param) const {
		this->get_override("execute")(sender, signal_code, param);
	}
};

// bs_imessaging wrapper
class bs_imessaging_pyw : public bs_imessaging, public wrapper< bs_imessaging > {
public:
	bool subscribe(int signal_code, const smart_ptr< bs_slot, true >& slot) const {
		return this->get_override("subscribe")(signal_code, slot);
	}

	bool unsubscribe(int signal_code, const smart_ptr< bs_slot, true >& slot) const {
		return this->get_override("unsubscribe")(signal_code, slot);
	}

	ulong num_slots(int signal_code) const {
		return this->get_override("num_slots")(signal_code);
	}

	bool fire_signal(int signal_code, const sp_obj& param) const {
		return this->get_override("fire_signal")(signal_code, param);
	}

	std::vector< int > get_signal_list() const {
		return this->get_override("get_signal_list")();
	}

	void dispose() const {
		delete this;
	}
};

void slot_tester(int, const sp_slot& slot) {
	slot->execute(NULL, 0, NULL);
	//bs_type_info ti = BS_GET_TI(*slot);
	//cout << &ti.get() << endl;
	//cout << ti.name() << endl;
}

}

// exporting function
void py_bind_messaging() {
	// export bs_slot wrapper
	class_<
		bs_slot_pyw,
		smart_ptr< bs_slot_pyw, true >,
		bases< bs_refcounter >,
		boost::noncopyable
		>
	("slot")
		.def("execute", pure_virtual(&bs_slot::execute))
	;
	// register smart_ptr conversions
	implicitly_convertible< smart_ptr< bs_slot_pyw >, sp_slot >();
	register_ptr_to_python< sp_slot >();

	// DEBUG
	def("slot_tester", &slot_tester);

	// bs_imessaging abstract class
	class_<
		bs_imessaging_pyw,
		smart_ptr< bs_imessaging_pyw, true >,
		bases< bs_refcounter >,
		boost::noncopyable
		>
	("imessaging")
		.def("subscribe", pure_virtual(&bs_imessaging::subscribe))
		.def("unsubscribe", pure_virtual(&bs_imessaging::unsubscribe))
		.def("num_slots", pure_virtual(&bs_imessaging::num_slots))
		.def("fire_signal", pure_virtual(&bs_imessaging::fire_signal))
		.def("get_signal_list", pure_virtual(&bs_imessaging::get_signal_list))
	;
	// register smart_ptr conversions
	implicitly_convertible< smart_ptr< bs_imessaging_pyw, true >, blue_sky::smart_ptr< bs_imessaging, true > >();
	register_ptr_to_python< blue_sky::smart_ptr< bs_imessaging, true > >();

	// bs_messaging
	class_<
		bs_messaging,
		smart_ptr< bs_messaging, true >,
		bases< bs_imessaging >,
		boost::noncopyable
		>
	("messaging", no_init)
		.def("subscribe", &bs_messaging::subscribe)
		.def("unsubscribe", &bs_messaging::unsubscribe)
		.def("num_slots", &bs_messaging::num_slots)
		.def("fire_signal", &bs_messaging::fire_signal)
		.def("get_signal_list", &bs_messaging::get_signal_list)
	;
	// register smart_ptr conversions
	implicitly_convertible< smart_ptr< bs_messaging, true >, blue_sky::smart_ptr< bs_imessaging, true > >();
}

}}	// eof namespace blue_sky::python

