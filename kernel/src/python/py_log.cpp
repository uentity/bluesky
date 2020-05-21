/// @file
/// @author uentity
/// @date 08.10.2018
/// @brief Logging subsystem Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/python/common.h>
#include <bs/log.h>
#include "../kernel/logging_subsyst.h"

#include <pybind11/chrono.h>
#include <pybind11/functional.h>

#include <spdlog/sinks/base_sink.h>
#include <spdlog/details/null_mutex.h>

NAMESPACE_BEGIN(blue_sky::python)
NAMESPACE_BEGIN()

using namespace blue_sky::log;
using logger = spdlog::logger;

///////////////////////////////////////////////////////////////////////////////
//  Custom Python sink that can be binded to any (or all) BS loggers
//
using printer_f = std::function< void (std::string, level_enum, spdlog::log_clock::time_point) >;

template<typename Mutex>
struct py_sink : public spdlog::sinks::base_sink<Mutex> {
	using base_t = spdlog::sinks::base_sink<Mutex>;

	py_sink(printer_f printer) : printer_(std::move(printer)) {}

protected:
	auto sink_it_(const spdlog::details::log_msg& msg) -> void override {
		spdlog::memory_buf_t formatted;
		base_t::formatter_->format(msg, formatted);
		printer_(fmt::to_string(formatted), msg.level, msg.time);
	}

	auto flush_() -> void override {}

private:
	printer_f printer_;
};

auto print_logger(logger& L, level_enum level, const py::args& args) -> std::string {
	std::string stape;
	for(const auto& arg : args) {
		if(!stape.empty()) stape += ' ';
		stape += py::str(arg);
	}
	L.log(level, "{}", stape);
	return stape;
}

// log channels available in BS by default
enum class Log { Out, Err };

inline auto print_r(Log channel, level_enum level, const py::args& args) {
	auto& bs_logger = channel == Log::Out ? bsout() : bserr();
	return print_logger(bs_logger.logger(), level, args);
};

inline auto print(Log channel, level_enum level, const py::args& args) {
	print_r(channel, level, args);
};

auto bind_log_impl(py::module& m) -> void {
	// log levels enum
	py::enum_<level_enum>(m, "Level")
		.value("info", level_enum::info)
		.value("warn", level_enum::warn)
		.value("err", level_enum::err)
		.value("critical", level_enum::critical)
		.value("trace", level_enum::trace)
		.value("debug", level_enum::debug)
		.value("off", level_enum::off)
	;

	// wrap spdlog::logger
	py::class_<logger>(m, "logger")
		.def_property_readonly("level", &logger::level)
		.def_property_readonly("name", &logger::name, py::return_value_policy::reference_internal)
		.def("should_log", &logger::should_log)
		.def("set_level", &logger::set_level)
		.def("set_pattern", [](logger& L, std::string pattern){ L.set_pattern(std::move(pattern)); })
		// main logging function
		.def("log", print_logger)
		// make overload with info default log level
		.def("log", [](logger& L, py::args args){
			print_logger(L, level_enum::info, args);
		})
	;

	// access BS loggers
	m.def("get_logger", &get_logger, py::return_value_policy::reference);

	m.def("set_custom_tag", &set_custom_tag, "tag"_a, "Include given tag to every BS log line");

	// register Python callback as custom BS logger sink
	m.def("register_sink", [](printer_f py_printer, const std::string& logger_name) {
		using namespace kernel::detail;
		// [NOTE] using null mutex, because locking is done via GIL
		auto s = std::make_shared<py_sink<spdlog::details::null_mutex>>(std::move(py_printer));
		logging_subsyst::add_custom_sink(s, logger_name);

		auto at_pyexit = py::module::import("atexit");
		at_pyexit.attr("register")(std::function< void() >{[ws = std::weak_ptr{s}] {
			// ensure we're switched to ST logging mode (all pending messages flushed
			{
				auto _ = py::gil_scoped_release{};
				logging_subsyst::toggle_mt_logs(false);
			}
			// remove added sink BEFORE interpreter is destructed
			logging_subsyst::remove_custom_sink(ws.lock());
		}});
	}, "print_cb"_a, "logger_name"_a = "", "If `logger_name` is empty, attach sink to all existing loggers");
}

NAMESPACE_END()

void py_bind_log(py::module& m) {
	auto logm = m.def_submodule("log", "BS logging");
	bind_log_impl(logm);

	///////////////////////////////////////////////////////////////////////////////
	//  `bs.print` family overloads
	//
	py::enum_<Log>(m, "Log")
		.value("Out", Log::Out)
		.value("Err", Log::Err)
	;

	// 1. most generic - print given log channel name + [level] + data -> return formatted string
	m.def("print_log_r", [](const char* name, level_enum level, py::args args) {
		return print_logger(get_logger(name), level, std::move(args));
	}, "channel_name"_a, "level"_a);
	// same as above, but formatted string isn't returned
	m.def("print_log", [](const char* name, level_enum level, py::args args) {
		print_logger(get_logger(name), level, std::move(args));
	}, "channel_name"_a, "level"_a);
	// if level is omitted -- print with info level
	m.def("print_log", [](const char* name, py::args args) {
		print_logger(get_logger(name), level_enum::info, std::move(args));
	}, "channel_name"_a);

	// 2. log channel is specified by enum
	m.def("print_r", &print_r, "channel"_a, "level"_a);
	m.def("print", &print, "channel"_a, "level"_a);
	// if level is omitted -- print info
	m.def("print", [](Log channel, const py::args& args) {
		print(channel, level_enum::info, args);
	}, "channel"_a);
	// if log channel is omitted -- print to bsout
	m.def("print", [](level_enum level, const py::args& args) {
		print(Log::Out, level, args);
	}, "level"_a);
	// if channel is omitted -- print to 'out'
	m.def("print", [](const py::args& args) {
		print(Log::Out, level_enum::info, args);
	});

	// 3. handy functions to print errors
	m.def("print_err", [](level_enum level, const py::args& args) {
		print(Log::Err, level, args);
	}, "level"_a);
	// if level is omitted -- print error to bserr channel
	m.def("print_err", [](const py::args& args) {
		print(Log::Err, level_enum::err, args);
	});
}

NAMESPACE_END(blue_sky::python)
