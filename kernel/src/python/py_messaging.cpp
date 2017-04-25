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

	void execute(sp_cobj sender, int signal_code, sp_obj param) const override {
		// Use modified version of PYBIND11_OVERLOAD_PURE macro code
		// original implementation would fail with runtime error that pure virtual method is called
		// if no overload was found. But slot should be safe in sutuation when Python object is
		// already destroyed. In such a case just DO NOTHING and return.
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const bs_slot *>(this), "execute");
		if (overload) {
			auto o = overload(std::move(sender), signal_code, std::move(param));
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		//PYBIND11_OVERLOAD_PURE(
		//	void,
		//	bs_slot,
		//	execute,
		//	std::move(sender), signal_code, std::move(param)
		//);
	}
};

// bs_imessaging wrapper
// template used to flatten trampoline hierarchy -- they don't support multiple inheritance
template<typename Next = bs_imessaging>
class py_bs_imessaging : public Next {
public:
	using Next::Next;

	bool subscribe(int signal_code, const sp_slot& slot) const override {
		PYBIND11_OVERLOAD_PURE(
			bool,
			Next,
			subscribe,
			signal_code, slot
		);
	}

	bool unsubscribe(int signal_code, const sp_slot& slot) const override {
		PYBIND11_OVERLOAD_PURE(
			bool,
			Next,
			unsubscribe,
			signal_code, slot
		);
	}

	ulong num_slots(int signal_code) const override {
		PYBIND11_OVERLOAD_PURE(
			ulong,
			Next,
			num_slots,
			signal_code
		);
	}

	bool fire_signal(int signal_code, const sp_obj& param, const sp_cobj& sender) const override {
		PYBIND11_OVERLOAD_PURE(
			bool,
			Next,
			fire_signal,
			signal_code, param, sender
		);
	}

	std::vector< int > get_signal_list() const override {
		PYBIND11_OVERLOAD_PURE(
			std::vector< int >,
			Next,
			get_signal_list
		);
	}
};

// resulting trampoline for bs_mesagins
class py_bs_messaging : public py_bs_imessaging< py_object<bs_messaging> > {
public:
	using py_bs_imessaging<py_object<bs_messaging>>::py_bs_imessaging;

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

	bool fire_signal(int signal_code, const sp_obj& param, const sp_cobj& sender) const override {
		PYBIND11_OVERLOAD(
			bool,
			bs_messaging,
			fire_signal,
			signal_code, param, sender
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
}

} //eof hidden namespace

// exporting function
void py_bind_messaging(py::module& m) {
	// export bs_slot wrapper
	py::class_<
		bs_slot, py_bs_slot, std::shared_ptr<bs_slot>
	>(m, "slot")
		.def(py::init<>())
		.def("execute", &bs_slot::execute)
	;

	// DEBUG
	m.def("slot_tester", &slot_tester);

	// bs_imessaging abstract class
	py::class_<
		bs_imessaging, py_bs_imessaging<>, std::shared_ptr<bs_imessaging>
	>(m, "imessaging")
		.def("subscribe"       , &bs_imessaging::subscribe)
		.def("unsubscribe"     , &bs_imessaging::unsubscribe)
		.def("num_slots"       , &bs_imessaging::num_slots)
		.def("fire_signal"     , &bs_imessaging::fire_signal,
			"signal_code"_a, "param"_a = nullptr, "sender"_a = nullptr
		)
		.def("get_signal_list" , &bs_imessaging::get_signal_list)
	;

	// bs_messaging
	bool (bs_messaging::*add_signal_ptr)(int) = &bs_messaging::add_signal;

	py::class_<
		bs_messaging, bs_imessaging, objbase, py_bs_messaging,
		std::shared_ptr< bs_messaging >
	>(m, "messaging", py::multiple_inheritance())
		BSPY_EXPORT_DEF(bs_messaging)
		BSPY_ENABLE_PYOBJ(bs_messaging)
		.def(py::init_alias<>())
		.def(py::init_alias< bs_messaging::sig_range_t >())
		.def("subscribe"       , &bs_messaging::subscribe)
		.def("unsubscribe"     , &bs_messaging::unsubscribe)
		.def("num_slots"       , &bs_messaging::num_slots)
		.def("fire_signal"     , &bs_messaging::fire_signal,
			"signal_code"_a, "param"_a = nullptr, "sender"_a = nullptr
		)
		.def("add_signal"      , add_signal_ptr)
		.def("remove_signal"   , &bs_messaging::remove_signal)
		.def("get_signal_list" , &bs_messaging::get_signal_list)
		.def("clear"           , &bs_messaging::clear)
		.def("test_slot1", [](const bs_messaging& src, const sp_slot& slot, const sp_obj param = nullptr) {
			slot->execute(src.shared_from_this(), 42, param);
		})
	;

	py_bind_signal(m);
}

NAMESPACE_END(python)
NAMESPACE_END(blue_sky)

