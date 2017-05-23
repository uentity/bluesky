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
#include <spdlog/sinks/null_sink.h>
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

// purpose of this function is to find log filename that is not locked by another process
spdlog::sink_ptr create_file_sink(const char* desired_fname) {
	fmt::MemoryWriter fname;
	for(ulong i = 0; i < 100; i++) {
		if(i)
			fname.write("{}_{}", desired_fname, i);
		else
			fname.write("{}", desired_fname);
		try {
			auto res = std::make_shared< spdlog::sinks::rotating_file_sink_mt >(fname.str(), 1024*1024*5, 1);
			if(res) return res;
		}
		catch(spdlog::spdlog_ex) {}
	}
	// in fail casr just return null sink
	return std::make_shared< spdlog::sinks::null_sink_st >();
}

}

spdlog::logger& kernel_logging_subsyst::get_log(const char* log_name) {
	struct logs_container {
		// construct and set default logs formaat
		logs_container() :
			logs({
				{"out", register_logger("out",
					std::make_shared< spdlog::sinks::stdout_sink_mt >(),
					create_file_sink("blue_sky.log")
				)},
				{"err", register_logger("err",
					std::make_shared< spdlog::sinks::stderr_sink_mt >(),
					create_file_sink("blue_sky_err.log")
			)}})
		{
			for(auto& log : logs) {
				log.second->set_pattern("[%Y-%m-%d %T.%e] [%L] %v");
			}
		}

		spdlog::logger& operator[](const char* name) {
			return *logs.at(name);
		}

		std::unordered_map< std::string, std::shared_ptr< spdlog::logger > > logs;
	};

	// let's create C++11 thread-safe singleton as static variable
	// static variable would be initialized only in one thread!
	// we need to use VS2015 to support this
	static logs_container logs;

	return logs[log_name];
}

