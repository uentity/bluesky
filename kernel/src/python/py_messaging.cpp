/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/bs.h>
#include <bs/compat/messaging.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(python)

void py_bind_signal(py::module&);

namespace {
using namespace std;

// bs_slot wrapper
class py_bs_slot : public bs_slot {
public:
	using bs_slot::bs_slot;

	void execute(const sp_mobj& sender, int signal_code, const sp_obj& param) const override {
		PYBIND11_OVERLOAD_PURE(
			void,
			bs_slot,
			execute,
			sender, signal_code, param
		);
	}
};

// bs_imessaging wrapper
class py_bs_imessaging : public bs_imessaging {
public:
	using bs_imessaging::bs_imessaging;

	bool subscribe(int signal_code, const sp_slot& slot) const override {
		PYBIND11_OVERLOAD_PURE(
			bool,
			bs_imessaging,
			subscribe,
			signal_code, slot
		);
	}

	bool unsubscribe(int signal_code, const sp_slot& slot) const override {
		PYBIND11_OVERLOAD_PURE(
			bool,
			bs_imessaging,
			unsubscribe,
			signal_code, slot
		);
	}

	ulong num_slots(int signal_code) const override {
		PYBIND11_OVERLOAD_PURE(
			ulong,
			bs_imessaging,
			num_slots,
			signal_code
		);
	}

	bool fire_signal(int signal_code, const sp_obj& param) const override {
		PYBIND11_OVERLOAD_PURE(
			bool,
			bs_imessaging,
			fire_signal,
			signal_code, param
		);
	}

	std::vector< int > get_signal_list() const override {
		PYBIND11_OVERLOAD_PURE(
			std::vector< int >,
			bs_imessaging,
			get_signal_list
		);
	}
};


class py_bs_messaging : public bs_messaging {
public:
	using bs_messaging::bs_messaging;

	bool subscribe(int signal_code, const sp_slot& slot) const override {
		PYBIND11_OVERLOAD(
			bool,
			bs_messaging,
			subscribe,
			signal_code, slot
		);
	}

	bool unsubscribe(int signal_code, const sp_slot& slot) const override {
		PYBIND11_OVERLOAD(
			bool,
			bs_messaging,
			unsubscribe,
			signal_code, slot
		);
	}

	ulong num_slots(int signal_code) const override {
		PYBIND11_OVERLOAD(
			ulong,
			bs_messaging,
			num_slots,
			signal_code
		);
	}

	bool fire_signal(int signal_code, const sp_obj& param) const override {
		PYBIND11_OVERLOAD(
			bool,
			bs_messaging,
			fire_signal,
			signal_code, param
		);
	}

	std::vector< int > get_signal_list() const override {
		PYBIND11_OVERLOAD(
			std::vector< int >,
			bs_messaging,
			get_signal_list
		);
	}

	bool add_signal(int signal_code) override {
		PYBIND11_OVERLOAD(
			bool,
			bs_messaging,
			add_signal,
			signal_code
		);
	}

	bool remove_signal(int signal_code) override {
		PYBIND11_OVERLOAD(
			bool,
			bs_messaging,
			remove_signal,
			signal_code
		);
	}
};

void slot_tester(int c, const sp_slot& slot, const sp_obj& param ) {
	slot->execute(nullptr, c, param);
	//bs_type_info ti = BS_GET_TI(*slot);
	//cout << &ti.get() << endl;
	//cout << ti.name() << endl;
}

}

// exporting function
void py_bind_messaging(py::module& m) {
	// export bs_slot wrapper
	py::class_<
		bs_slot, py_bs_slot,
		std::shared_ptr< bs_slot >
	>(m, "slot")
		.def(py::init<>())
		.def("execute", &bs_slot::execute)
	;

	// DEBUG
	m.def("slot_tester", &slot_tester);

	// bs_imessaging abstract class
	py::class_<
		bs_imessaging, py_bs_imessaging,
		std::shared_ptr< bs_imessaging >
	>(m, "imessaging")
		.def("subscribe"       , &bs_imessaging::subscribe)
		.def("unsubscribe"     , &bs_imessaging::unsubscribe)
		.def("num_slots"       , &bs_imessaging::num_slots)
		.def("fire_signal"     , &bs_imessaging::fire_signal)
		.def("get_signal_list" , &bs_imessaging::get_signal_list)
	;

	// bs_messaging
	bool (bs_messaging::*add_signal_ptr)(int) = &bs_messaging::add_signal;
	py::class_<
		bs_messaging, py_bs_messaging,
		std::shared_ptr< bs_messaging >
	>(m, "messaging")
		.def(py::init<>())
		.def(py::init< bs_messaging::sig_range_t >())
		.def("subscribe"       , &bs_messaging::subscribe)
		.def("unsubscribe"     , &bs_messaging::unsubscribe)
		.def("num_slots"       , &bs_messaging::num_slots)
		.def("fire_signal"     , &bs_messaging::fire_signal)
		.def("add_signal"      , add_signal_ptr)
		.def("remove_signal"   , &bs_messaging::remove_signal)
		.def("get_signal_list" , &bs_messaging::get_signal_list)
		.def("clear"           , &bs_messaging::clear)
	;

	py_bind_signal(m);
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

