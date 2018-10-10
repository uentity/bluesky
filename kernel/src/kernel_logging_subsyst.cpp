/// @file
/// @author uentity
/// @date 30.08.2016
/// @brief Kernel logging subsystem impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include <bs/kernel_errors.h>
#include <bs/log.h>
#include "kernel_logging_subsyst.h"

#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <unordered_map>
#include <atomic>

#define FILE_LOG_PATTERN "[%Y-%m-%d %T.%e] [%L] %v"
#define CONSOLE_LOG_PATTERN "[%L] %v"
#define OUT_FNAME_DEFAULT "blue_sky.log"
#define ERR_FNAME_DEFAULT "blue_sky_err.log"
constexpr auto ROTATING_FSIZE_DEFAULT = 1024*1024*10;
constexpr auto DEF_FLUSH_INTERVAL = std::chrono::seconds(5);
constexpr auto DEF_FLUSH_LEVEL = spdlog::level::err;

using namespace blue_sky::detail;
using namespace blue_sky;

namespace {
// [TODO] collect spdlog tune code in one place instead of distributing it here and there
// Can be done after config subsystem is ready

///////////////////////////////////////////////////////////////////////////////
//  log-related globals
//
// fallback to null sink
const auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
// fallback null st logger
// + init default flush level
const auto null_st_logger = [] {
	// set global minimum flush level
	spdlog::flush_on(DEF_FLUSH_LEVEL);
	return spdlog::create<spdlog::sinks::null_sink_mt>("null");
}();

// flag that indicates whether we have switched to mt logs
std::atomic<bool> are_logs_mt(false);

///////////////////////////////////////////////////////////////////////////////
//  create sinks
//
// purpose of this function is to find log filename that is not locked by another process
spdlog::sink_ptr create_file_sink(const char* desired_fname) {
	static std::unordered_map<std::string, spdlog::sink_ptr> sinks_;
	static std::mutex solo_;

	// protect sinks_ map from mt-access
	std::lock_guard<std::mutex> play_solo(solo_);

	// check if we already created sink with desired fname
	auto S = sinks_.find(desired_fname);
	if(S != sinks_.end())
		return S->second;

	// if not -- create it
	fmt::memory_buffer fname;
	for(int i = 0; i < 100; i++) {
		fname.clear();
		if(i)
			fmt::format_to(fname, "{}_{}", i, desired_fname);
		else
			fmt::format_to(fname, "{}", desired_fname);
		try {
			auto res = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
				fmt::to_string(fname), ROTATING_FSIZE_DEFAULT, 1
			);
			if(res) {
				res->set_pattern(FILE_LOG_PATTERN);
				sinks_.insert( {desired_fname, res} );
				return res;
			}
		}
		catch(...) {}
	}
	// can't create log file - return null sink
	return null_sink;
}

template<typename Sink>
auto create_console_sink() -> spdlog::sink_ptr {
	static const auto sink_ = []() -> spdlog::sink_ptr {
		try {
			auto S = std::make_shared<Sink>();
			if(S) {
				S->set_pattern(CONSOLE_LOG_PATTERN);
				return S;
			}
		}
		catch(...) {}
		return null_sink;
	}();

	return sink_;
}

///////////////////////////////////////////////////////////////////////////////
//  create loggers
//
template< typename... Sinks >
auto create_logger(const char* log_name, Sinks... sinks) -> std::shared_ptr<spdlog::logger> {
	spdlog::sink_ptr S[] = { sinks... };
	try {
		auto L = std::make_shared<spdlog::logger>( log_name, std::begin(S), std::end(S) );
		spdlog::register_logger(L);
		return L;
	}
	//catch(const spdlog::spdlog_ex&) {}
	catch(...) {}
	return null_st_logger;
}

template< typename... Sinks >
auto create_async_logger(const char* log_name, Sinks... sinks) -> std::shared_ptr<spdlog::logger> {
	// create null logger and ensure thread pool is initialized
	static const auto null_mt_logger = spdlog::create_async_nb<spdlog::sinks::null_sink_mt>("null");

	spdlog::sink_ptr S[] = { sinks... };
	try {
		auto L = std::make_shared<spdlog::async_logger>(
			log_name, std::begin(S), std::end(S), spdlog::thread_pool()
		);
		spdlog::register_logger(L);
		return L;
	}
	//catch(const spdlog::spdlog_ex&) {}
	catch(...) {}
	return null_mt_logger;
}

// global bs_log loggers for "out" and "err"
auto bs_out_instance = std::make_unique<log::bs_log>("out"),
	 bs_err_instance = std::make_unique<log::bs_log>("err");

} // hidden namespace

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(log)

/*-----------------------------------------------------------------------------
 *  get/create spdlog::logger backend for bs_log
 *-----------------------------------------------------------------------------*/
auto get_logger(const char* log_name) -> spdlog::logger& {
	using f = std::shared_ptr<spdlog::logger> (*)();
	// setup static map of known single-thread loggers generators
	static const std::unordered_map<std::string, f> st_log_gen {
		{"out", [] { return create_logger("out",
			create_console_sink<spdlog::sinks::stdout_sink_mt>(), create_file_sink(OUT_FNAME_DEFAULT)
		); }},
		{"err", [] { return create_logger("err",
			create_console_sink<spdlog::sinks::stderr_sink_mt>(), create_file_sink(ERR_FNAME_DEFAULT)
		); }}
	};
	// and multithreaded ones
	static const std::unordered_map<std::string, f> mt_log_gen {
		{"out", [] { return create_async_logger("out",
			create_console_sink<spdlog::sinks::stdout_sink_mt>(), create_file_sink(OUT_FNAME_DEFAULT)
		); }},
		{"err", [] { return create_async_logger("err",
			create_console_sink<spdlog::sinks::stderr_sink_mt>(), create_file_sink(ERR_FNAME_DEFAULT)
		); }}
	};

	// if log already registered -- return it
	if(auto L = spdlog::get(log_name))
		return *L;
	// otherwise create it
	// switch to one of these maps depending on `use_mt_logs` flag
	const auto& log_gen = are_logs_mt ? mt_log_gen : st_log_gen;
	auto pgen = log_gen.find(log_name);
	return pgen == log_gen.end() ? *null_st_logger : *pgen->second();
}

NAMESPACE_END(log)

// switch between mt- and st- logs
auto kernel_logging_subsyst::toggle_mt_logs(bool turn_on) -> void {
	if(are_logs_mt.exchange(turn_on) != turn_on) {
		// drop all previousely created logs
		spdlog::drop_all();
		// and create new ones
		bs_out_instance = std::make_unique<log::bs_log>("out");
		bs_err_instance = std::make_unique<log::bs_log>("err");
		// setup periodic flush
		spdlog::flush_every(DEF_FLUSH_INTERVAL);
	}
}

/*-----------------------------------------------------------------
 * access to main log channels
 *----------------------------------------------------------------*/
log::bs_log& bsout() {
	return *bs_out_instance;
}

log::bs_log& bserr() {
	return *bs_err_instance;
}

NAMESPACE_END(blue_sky)

