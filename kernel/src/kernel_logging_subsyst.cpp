/// @file
/// @author uentity
/// @date 30.08.2016
/// @brief Kernel logging subsystem impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel.h>
#include <bs/error.h>
#include <bs/kernel_errors.h>
#include <bs/log.h>
#include "kernel_logging_subsyst.h"

#include <spdlog/async.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <boost/filesystem.hpp>

#include <unordered_map>
#include <atomic>

#define BSCONFIG BS_KERNEL.config()
#define FILE_LOG_PATTERN "[%Y-%m-%d %T.%e] [%L] %v"
#define CONSOLE_LOG_PATTERN "[%L] %v"
#define OUT_FNAME_DEFAULT "blue_sky.log"
#define ERR_FNAME_DEFAULT "blue_sky_err.log"
constexpr auto ROTATING_FSIZE_DEFAULT = 1024*1024*10;
constexpr auto DEF_FLUSH_INTERVAL = 5;
constexpr auto DEF_FLUSH_LEVEL = spdlog::level::err;

using namespace blue_sky::detail;
using namespace blue_sky;
namespace bfs = boost::filesystem;

NAMESPACE_BEGIN()
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
// if `logger_name` is set -- tune sink with configured values
spdlog::sink_ptr create_file_sink(const std::string& desired_fname, const std::string& logger_name = "") {
	static std::unordered_map<std::string, spdlog::sink_ptr> sinks_;
	static std::mutex solo_;

	// protect sinks_ map from mt-access
	std::lock_guard<std::mutex> play_solo(solo_);

	spdlog::sink_ptr res;
	// check if we already created sink with desired fname
	auto S = sinks_.find(desired_fname);
	if(S != sinks_.end()) {
		res = S->second;
	}
	else {
		// if not -- create it
		// split desired fname into name and extension
		const bfs::path logf(desired_fname);
		const auto logf_ext = logf.extension();
		const auto logf_body = logf.parent_path() / logf.stem();
		// create parent dir
		if(!bfs::exists(logf.parent_path())) {
			boost::system::error_code er;
			bfs::create_directories(logf.parent_path(), er);
			if(er) {
				std::cerr << "[E] Failed to create log file " << desired_fname
					<< ": " << er.message() << std::endl;
				return null_sink;
			}
		}

		for(int i = 0; i < 100; i++) {
			auto cur_logf = logf_body;
			if(i)
				cur_logf += std::string("_") + std::to_string(i);
			cur_logf += logf_ext;

			try {
				res = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
					cur_logf.string(), ROTATING_FSIZE_DEFAULT, 1
				);
				if(res) {
					res->set_pattern(FILE_LOG_PATTERN);
					sinks_.insert( {desired_fname, res} );
					break;
				}
			}
			catch(...) {}
		}
	}

	// configure sink
	if(res && !logger_name.empty()) {
		res->set_pattern(caf::get_or(
			BSCONFIG, std::string("logger.") + logger_name + "-file-format",
			FILE_LOG_PATTERN
		));
	}

	// can't create log file - return null sink
	return res ? res : null_sink;
}

// if `logger_name` is set -- tune sink with configured values
template<typename Sink>
auto create_console_sink(const std::string& logger_name = "") -> spdlog::sink_ptr {
	static const auto sink_ = []() -> spdlog::sink_ptr {
		spdlog::sink_ptr S = null_sink;
		try {
			if((S = std::make_shared<Sink>()))
				S->set_pattern(CONSOLE_LOG_PATTERN);
		}
		catch(...) {}
		return S;
	}();

	if(!logger_name.empty()) {
		sink_->set_pattern(caf::get_or(
			BSCONFIG, std::string("logger.console-") + logger_name + "-format",
			CONSOLE_LOG_PATTERN
		));
	}

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
		L->flush_on(static_cast<spdlog::level::level_enum>(caf::get_or(
			BSCONFIG, std::string("logger.") + log_name + "-flush-level",
			std::uint8_t(DEF_FLUSH_LEVEL)
		)));
		spdlog::register_logger(L);
		return L;
	}
	catch(...) {}
	return null_mt_logger;
}

// global bs_log loggers for "out" and "err"
auto bs_out_instance = std::make_unique<log::bs_log>("out"),
	 bs_err_instance = std::make_unique<log::bs_log>("err");

NAMESPACE_END() // hidden namespace

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(log)

/*-----------------------------------------------------------------------------
 *  get/create spdlog::logger backend for bs_log
 *-----------------------------------------------------------------------------*/
auto get_logger(const char* log_name) -> spdlog::logger& {
	using f = std::shared_ptr<spdlog::logger> (*)();

	// setup static map of known single-thread loggers generators
	static const std::unordered_map<std::string, f> st_log_gen {
		{"out", [] { return create_logger("out",
			create_console_sink<spdlog::sinks::stdout_sink_mt>()
			//create_file_sink(caf::get_or(BSCONFIG, "logger.out-file-name", OUT_FNAME_DEFAULT))
		); }},
		{"err", [] { return create_logger("err",
			create_console_sink<spdlog::sinks::stderr_sink_mt>()
			//create_file_sink(caf::get_or(BSCONFIG, "logger.err-file-name", ERR_FNAME_DEFAULT))
		); }}
	};
	// and multithreaded ones
	static const std::unordered_map<std::string, f> mt_log_gen {
		{"out", [] {
			return create_async_logger("out",
				create_console_sink<spdlog::sinks::stdout_sink_mt>("out"),
				create_file_sink(caf::get_or(BSCONFIG, "logger.out-file-name", OUT_FNAME_DEFAULT), "out")
			);
		}},
		{"err", [] {
			return create_async_logger("err",
				create_console_sink<spdlog::sinks::stderr_sink_mt>("err"),
				create_file_sink(caf::get_or(BSCONFIG, "logger.err-file-name", ERR_FNAME_DEFAULT), "err")
			);
		}}
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
		spdlog::flush_every(std::chrono::seconds(caf::get_or(
			BSCONFIG, "logger.flush-interval", DEF_FLUSH_INTERVAL
		)));
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

