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

BS_API const sp_channel &bs_end2(const sp_channel &r) {
		r.lock()->set_can_output(true);
		r.lock()->send_to_subscribers();
		return r;
}

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
	if (auto_newline)
		*c.lock() << str << bs_end;
	else
		bs_end2(*c.lock() << str);
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

void py_bs_channel::set_wait_end() const {
	c.lock()->set_wait_end();
}

void py_bs_channel::set_auto_newline(bool nl = false) {
	auto_newline = nl;
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

	class_<stream_wrapper, noncopyable>("stream", init<const boost::python::object&>())
		.def("write", &stream_wrapper::write);

	class_<py_stream, noncopyable>("wstream", init<const boost::python::object&>());

	class_<py_bs_channel>("channel", init <const std::string &> ())
		.def(init<std::string>())
		.def("attach",&py_bs_channel::attach)
		.def("detach",&py_bs_channel::detach)
		.def("get_name",&py_bs_channel::get_name)
		.def("write",&py_bs_channel::write)
		.def("set_output_time",&py_bs_channel::set_output_time)
		.def("set_wait_end",&py_bs_channel::set_wait_end)
		.def("set_auto_newline",&py_bs_channel::set_auto_newline);

	enum_<bs_log::signal_codes>("log_signal_codes")
		.value("log_channel_added",bs_log::log_channel_added)
		.value("log_channel_removed",bs_log::log_channel_removed)
		.export_values();
}

}	//namespace blue_sky::python
}	//namespace blue_sky
