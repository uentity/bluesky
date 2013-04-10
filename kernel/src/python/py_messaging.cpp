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

#include "bs_messaging.h"
#include "bs_object_base.h"
#include "py_bs_exports.h"
#include "py_smart_ptr.h"
#include <boost/python.hpp>

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

class bs_messaging_pyw : public bs_messaging, public wrapper< bs_messaging > {
public:
	bs_messaging_pyw(const bs_messaging& rhs)
		: bs_messaging(rhs)
	{}
	bs_messaging_pyw()
	{}

	bool subscribe(int signal_code, const sp_slot& slot) const {
		if(override f = this->get_override("subscribe"))
			return f(signal_code, slot);
		return bs_messaging::subscribe(signal_code, slot);
	}
	bool def_subscribe(int signal_code, const sp_slot& slot) const {
		return this->subscribe(signal_code, slot);
	}

	bool unsubscribe(int signal_code, const sp_slot& slot) const {
		if(override f = this->get_override("unsubscribe"))
			return f(signal_code, slot);
		return bs_messaging::unsubscribe(signal_code, slot);
	}
	bool def_unsubscribe(int signal_code, const sp_slot& slot) const {
		return this->unsubscribe(signal_code, slot);
	}

	ulong num_slots(int signal_code) const {
		if(override f = this->get_override("num_slots"))
			return f(signal_code);
		return bs_messaging::num_slots(signal_code);
	}
	ulong def_num_slots(int signal_code) const {
		return this->num_slots(signal_code);
	}

	bool fire_signal(int signal_code, const sp_obj& param = sp_obj (NULL)) const {
		if(override f = this->get_override("fire_signal"))
			return f(signal_code, param);
		return bs_messaging::fire_signal(signal_code, param);
	}
	bool def_fire_signal(int signal_code, const sp_obj& param = sp_obj (NULL)) const {
		return this->fire_signal(signal_code, param);
	}

	bool add_signal(int signal_code) {
		if(override f = this->get_override("add_signal"))
			return f(signal_code);
		return bs_messaging::add_signal(signal_code);
	}
	bool def_add_signal(int signal_code) {
		return this->add_signal(signal_code);
	}

	bool remove_signal(int signal_code) {
		if(override f = this->get_override("remove_signal"))
			return f(signal_code);
		return bs_messaging::remove_signal(signal_code);
	}
	bool def_remove_signal(int signal_code) {
		return this->remove_signal(signal_code);
	}
};

void slot_tester(int, const sp_slot& slot) {
	slot->execute(NULL, 0, NULL);
	//bs_type_info ti = BS_GET_TI(*slot);
	//cout << &ti.get() << endl;
	//cout << ti.name() << endl;
}

}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(connect_overl, connect, 1, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(fire_overl, fire, 0, 2)

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
	bool (bs_messaging::*add_signal_ptr)(int) = &bs_messaging::add_signal;
	class_<
		bs_messaging_pyw,
		smart_ptr< bs_messaging_pyw, true >,
		bases< bs_imessaging >
		>
	("messaging")
		.def("subscribe"     , &bs_messaging::subscribe     , &bs_messaging_pyw::def_subscribe)
		.def("unsubscribe"   , &bs_messaging::unsubscribe   , &bs_messaging_pyw::def_unsubscribe )
		.def("num_slots"     , &bs_messaging::num_slots     , &bs_messaging_pyw::def_num_slots)
		.def("fire_signal"   , &bs_messaging::fire_signal   , &bs_messaging_pyw::def_fire_signal)
		.def("add_signal"    , add_signal_ptr               , &bs_messaging_pyw::def_add_signal)
		.def("remove_signal" , &bs_messaging::remove_signal , &bs_messaging_pyw::def_remove_signal)
		.def("get_signal_list", &bs_messaging::get_signal_list)
	;
	// register smart_ptr conversions
	implicitly_convertible< smart_ptr< bs_messaging_pyw, true >, blue_sky::smart_ptr< bs_messaging, true > >();
	implicitly_convertible< smart_ptr< bs_messaging, true >, blue_sky::smart_ptr< bs_imessaging, true > >();
	register_ptr_to_python< blue_sky::smart_ptr< bs_messaging, true > >();

	// bs_signal
	class_<
		bs_signal,
		smart_ptr< bs_signal, true >,
		bases< bs_refcounter >
		>
	("signal", init< int >())
		.def("init", &bs_signal::init)
		.def_readonly("code", &bs_signal::get_code)
		.def("connect", &bs_signal::connect, connect_overl())
		.def("disconnect", &bs_signal::disconnect)
		.def_readonly("num_slots", &bs_signal::num_slots)
		.def("fire", &bs_signal::fire, fire_overl())
	;
}

}}	// eof namespace blue_sky::python

