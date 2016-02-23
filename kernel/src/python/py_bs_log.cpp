/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "py_bs_log.h"
#include "py_bs_exports.h"
#include "py_bs_object_base.h"
#include "bs_kernel.h"

#include <bs_exception.h>
#include <boost/python/call_method.hpp>
#include <boost/python/enum.hpp>

using namespace boost::python;

namespace blue_sky {
namespace python {

py_bs_log::py_bs_log()
	: //py_bs_messaging(sp_log(new bs_log(give_log::Instance())))
	l(BS_KERNEL.get_log ())
{}

//py_bs_channel *py_bs_log::get(const std::string &name_) const {
//	return new py_bs_channel(l[name_]);
//}
//
//const sp_channel& py_bs_log::operator[](const std::string &name_) const {
//	return l[name_];
//}

py_bs_channel py_bs_log::add_channel(const py_bs_channel &ch) {
	return py_bs_channel(l.add_channel(ch.c));
}

bool py_bs_log::rem_channel(const std::string &ch_name) {
	return l.rem_channel(ch_name);
}

void stream_wrapper::write(const std::string &str) const {
	boost::python::call_method<void>(obj.ptr(), "write", str);
}

// void py_stream::write(const std::string &str) const {
// 	try {
// 		if (override f = this->get_override("write"))
// 			f(str);
// 		else
// 			BSERROR << "Function write does not exists, or is not correct!" << bs_end;
// 	}
// 	catch(...) {
// 		throw bs_exception("py_stream","May be member function \"void write(const std::string&) const\" is not overriden!");
// 	}
// }

py_bs_channel::py_bs_channel(const std::string &a) : c(new bs_channel(a)),auto_newline(true) {}
py_bs_channel::py_bs_channel(const sp_channel &s) : c(s),auto_newline(true) {}

void py_bs_channel::write(const char *str) const {
  locked_channel (c, __FILE__, __LINE__) << str << bs_end;
}

bool py_bs_channel::attach(const py_stream &s) const {
	return c.lock()->attach(s.spstream);
}

bool py_bs_channel::detach(const py_stream &s) const {
	return c.lock()->detach(s.spstream);
}

std::string py_bs_channel::get_name() const {
	return c.lock()->get_name();
}

void py_bs_channel::set_output_time() const {
	c.lock()->set_output_time();
}

py_thread_log::py_thread_log() : l(BS_KERNEL.get_tlog ()) {}

py_bs_channel py_thread_log::add_log_channel(const std::string &name) {
	return py_bs_channel(l.add_log_channel(name));
}

bool py_thread_log::add_log_stream(const std::string &ch_name, const py_stream &pstream) {
	return l.add_log_stream(ch_name,pstream.spstream);
}

bool py_thread_log::rem_log_channel(const std::string &ch_name) {
	return l.rem_log_channel(ch_name);
}

bool py_thread_log::rem_log_stream(const std::string &ch_name, const py_stream &pstream) {
	return l.rem_log_stream(ch_name,pstream.spstream);
}

//const py_bs_channel *py_thread_log::get(const std::string &ch_name) const {
//	return new py_bs_channel(l[ch_name]);
//}

//const sp_channel &py_thread_log::operator[](const std::string &ch_name) const {
//	return l[ch_name];
//}

bool py_bs_log::subscribe(int signal_code, const python_slot& slot) const {
	return l.subscribe(signal_code,slot.spslot);
}

bool py_bs_log::unsubscribe(int signal_code, const python_slot& slot) const {
	return l.unsubscribe(signal_code,slot.spslot);
}

ulong py_bs_log::num_slots(int signal_code) const {
	return l.num_slots(signal_code);
}

bool py_bs_log::fire_signal(int signal_code, const py_objbase* param) const {
	return l.fire_signal(signal_code,param->sp);
}

std::vector< int > py_bs_log::get_signal_list() const {
	return l.get_signal_list();
}

//std::list< std::string > py_bs_log::get_ch_list() const {
//	return l.channel_list();
//}

void py_export_log() {
	class_<py_bs_log, /*bases<py_bs_messaging>,*/ noncopyable>("log")
		.def("add_channel",&py_bs_log::add_channel)
		.def("rem_channel",&py_bs_log::rem_channel)
		//.def("get",&py_bs_log::get, return_value_policy<manage_new_object>())
		.def("subscribe",&py_bs_log::subscribe)
		.def("unsubscribe",&py_bs_log::unsubscribe)
		.def("num_slots",&py_bs_log::num_slots)
		.def("fire_signal",&py_bs_log::fire_signal)
		.def("get_signal_list",&py_bs_log::get_signal_list)
		//.def("get_ch_list",&py_bs_log::get_ch_list)
	;

	class_<py_thread_log, noncopyable>("thread_log")
		.def("add_log_channel",&py_thread_log::add_log_channel)
		.def("rem_log_channel",&py_thread_log::rem_log_channel)
		.def("add_log_stream",&py_thread_log::add_log_stream)
		.def("rem_log_stream",&py_thread_log::rem_log_stream)
		//.def("get",&py_thread_log::get, return_value_policy<manage_new_object>())
    ;

	//class_<py_stream, noncopyable>("stream")
		//.def("write",pure_virtual(&bs_stream::write));

	class_<stream_wrapper, noncopyable>("stream", init<const std::string &, const boost::python::object&>())
		.def("write", &stream_wrapper::write);

	class_<py_stream, noncopyable>("wstream", init<const std::string &, const boost::python::object&>());

	class_<py_bs_channel>("channel", init <const std::string &> ())
		.def(init<std::string>())
		.def("attach",&py_bs_channel::attach)
		.def("detach",&py_bs_channel::detach)
		.def("get_name",&py_bs_channel::get_name)
		.def("write",&py_bs_channel::write)
		.def("set_output_time",&py_bs_channel::set_output_time)
    ;

	enum_<bs_log::signal_codes>("log_signal_codes")
		.value("log_channel_added",bs_log::log_channel_added)
		.value("log_channel_removed",bs_log::log_channel_removed)
		.export_values();
}

}	//namespace blue_sky::python
}	//namespace blue_sky
