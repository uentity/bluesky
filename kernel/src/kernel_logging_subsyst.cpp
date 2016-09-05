/// @file
/// @author uentity
/// @date 30.08.2016
/// @brief Kernel logging subsystem impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "kernel_logging_subsyst.h"
#include <spdlog/spdlog.h>
#include <unordered_map>

using namespace blue_sky::detail;

namespace {

template< typename... Sinks >
auto register_logger(const char* log_name, Sinks... sinks) {
	spdlog::sink_ptr S[] = { sinks... };
	std::shared_ptr< spdlog::logger > L(std::make_shared< spdlog::logger >(
		log_name, begin(S), end(S)
	));
	spdlog::register_logger(L);
	return L;
}

}

spdlog::logger& kernel_logging_subsyst::get_log(const char* log_name) {
	// let's create C++11 thread-safe singleton as static variable
	// static variable would be initialized only in one thread!
	// we need to use VS2015 to support this
	static std::unordered_map< std::string, std::shared_ptr< spdlog::logger > > logs({
		{"out", register_logger("out",
			std::make_shared< spdlog::sinks::stdout_sink_mt >(),
			std::make_shared< spdlog::sinks::rotating_file_sink_mt >("blue_sky", "log", 1024*1024*5, 1)
		)},
		{"err", register_logger("err",
			std::make_shared< spdlog::sinks::stderr_sink_mt >(),
			std::make_shared< spdlog::sinks::rotating_file_sink_mt >("blue_sky_err", "log", 1024*1024*5, 1)
	)}});

	return *logs.at(log_name);
}

