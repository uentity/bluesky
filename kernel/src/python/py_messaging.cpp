/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Python bindings for BS signal-slot subsystem
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/python/common.h>
#include <bs/compat/messaging.h>

BSPY_ANY_CAST_EXTRA(long double)
#include <bs/python/any.h>

NAMESPACE_BEGIN(blue_sky::python)
NAMESPACE_BEGIN()
using namespace std;

// bs_slot wrapper
class py_bs_slot : public bs_slot {
public:
	using bs_slot::bs_slot;

	void execute(std::any sender, int signal_code, std::any param) const override {
		// Use modified version of PYBIND11_OVERLOAD_PURE macro code
		// original implementation would fail with runtime error that pure virtual method is called
		// if no overload was found. But slot should be safe in sutuation when Python object is
		// already destroyed. In such a case just DO NOTHING and return.
		pybind11::gil_scoped_acquire gil;
		pybind11::function override = pybind11::get_override(static_cast<const bs_slot *>(this), "execute");
		if(override) {
			auto o = override(std::move(sender), signal_code, std::move(param));
			if(pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
	}
};

NAMESPACE_END()

// exporting function
void py_bind_messaging(py::module& m) {
	// slot
	py::class_<
		bs_slot, py_bs_slot, std::shared_ptr<bs_slot>
	>(m, "slot")
		.def(py::init<>())
		.def("execute", &bs_slot::execute)
	;

	// signal
	py::class_<
		bs_signal,
		std::shared_ptr< bs_signal >
	>(m, "signal")
		.def(py::init<int>())
		.def("init", &bs_signal::init)
		.def_property_readonly("get_code", &bs_signal::get_code)
		.def("connect", &bs_signal::connect, "slot"_a, "sender"_a = nullptr)
		.def("disconnect", &bs_signal::disconnect)
		.def_property_readonly("num_slots", &bs_signal::num_slots)
		.def("fire", &bs_signal::fire, "sender"_a = nullptr, "param"_a = std::any{})
	;

	// bs_messaging
	bool (bs_messaging::*add_signal_ptr)(int) = &bs_messaging::add_signal;

	py::class_<
		bs_messaging, objbase,
		std::shared_ptr<bs_messaging>
	>(m, "messaging", py::multiple_inheritance())
		BSPY_EXPORT_DEF(bs_messaging)

		.def(py::init<>())
		.def(py::init<bs_messaging::sig_range_t>())
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
	;
}

NAMESPACE_END(blue_sky::python)
