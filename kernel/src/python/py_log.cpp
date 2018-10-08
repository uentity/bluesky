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

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(python)
namespace {
using namespace blue_sky::log;
using logger = spdlog::logger;

auto print_logger(logger& L, level_enum level, const py::args& args) -> void {
	std::string stape;
	for(const auto& arg : args) {
		if(!stape.empty()) stape += ' ';
		stape += py::str(arg);
	}
	L.log(level, "{}", stape);
}

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
	py::class_<logger, std::shared_ptr<logger>>(m, "logger")
		.def("should_log", &logger::should_log)
		.def("set_level", &logger::set_level)
		.def("set_pattern", [](logger& L, std::string pattern){ L.set_pattern(pattern); })
		.def_property_readonly("level", &logger::level)
		.def_property_readonly("name", &logger::name, py::return_value_policy::reference_internal)
		// main logging function
		.def("log", print_logger)
		// make overload with info default log level
		.def("log", [](logger& L, py::args args){
			print_logger(L, level_enum::info, args);
		})
	;

	// access BS loggers
	m.def("get_logger", &get_logger, py::return_value_policy::reference);
}

} // eof hidden namespace

void py_bind_log(py::module& m) {
	auto logm = m.def_submodule("log", "BS logging");
	bind_log_impl(logm);

	///////////////////////////////////////////////////////////////////////////////
	//  `bs.print` family overloads
	//
	// log channels available in BS by default
	enum class Log { Out, Err };
	py::enum_<Log>(m, "Log")
		.value("Out", Log::Out)
		.value("Err", Log::Err)
	;

	static const auto print = [](Log channel, level_enum level, const py::args& args) {
		// [TODO] fast, but non-extensible code
		const auto name = channel == Log::Out ? "out" : "err";
		print_logger(get_logger(name), level, args);
	};

	// 1. most generic - print given log channel name + [level] + data
	m.def("print_log", [](const char* name, level_enum level, py::args args) {
		print_logger(get_logger(name), level, std::move(args));
	});
	// if level is omitted -- print with info level
	m.def("print_log", [](const char* name, py::args args) {
		print_logger(get_logger(name), level_enum::info, std::move(args));
	});

	// 2. log channel is specified by enum
	m.def("print", print);
	// if level is omitted -- print info
	m.def("print", [](Log channel, const py::args& args) {
		print(channel, level_enum::info, args);
	});
	// if channel is omitted -- print to 'out'
	m.def("print", [](const py::args& args) {
		print(Log::Out, level_enum::info, args);
	});

	// 3. handy functions to print errors
	m.def("print_err", [](level_enum level, const py::args& args) {
		print(Log::Err, level, args);
	});
	// if level is omitted -- print info to error channel
	m.def("print_err", [](const py::args& args) {
		print(Log::Err, level_enum::info, args);
	});
}

NAMESPACE_END(python) NAMESPACE_END(blue_sky)

