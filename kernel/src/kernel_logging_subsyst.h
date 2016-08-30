/// @file
/// @author uentity
/// @date 24.08.2016
/// @brief Create and register BlueSky logs
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <bs/common.h>
#include <spdlog/spdlog.h>
#include <unordered_map>

// hide implementation
namespace blue_sky { namespace {

struct kernel_logging_subsyst {
	std::unordered_map< std::string, std::shared_ptr< spdlog::logger > > logs_;
	//std::shared_ptr< spdlog::logger > bs_out;
	//std::shared_ptr< spdlog::logger > bs_err;

	template< typename... Sinks >
	auto register_logger(const char* log_name, Sinks... sinks) {
		spdlog::sink_ptr S[] = { sinks... };
		std::shared_ptr< spdlog::logger > L(std::make_shared< spdlog::logger >(
			log_name, begin(S), end(S)
		));
		spdlog::register_logger(L);
		return L;
	}

	kernel_logging_subsyst() :
		logs_({
		{"out", register_logger("out",
			std::make_shared< spdlog::sinks::stdout_sink_mt >(),
			std::make_shared< spdlog::sinks::rotating_file_sink_mt >("blue_sky", "log", 1024*1024*5, 1)
		)},
		{"err", register_logger("err",
			std::make_shared< spdlog::sinks::stderr_sink_mt >(),
			std::make_shared< spdlog::sinks::rotating_file_sink_mt >("blue_sky_err", "log", 1024*1024*5, 1)
		)}})
	{}

	spdlog::logger& get_log(const char* log_name = "out") {
		return *logs_.at(log_name);
	}
};
	
}} /* namespace blue_sky */

