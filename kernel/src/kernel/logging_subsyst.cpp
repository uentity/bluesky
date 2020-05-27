/// @file
/// @author uentity
/// @date 30.08.2016
/// @brief Kernel logging subsystem impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "logging_subsyst.h"

#include <bs/log.h>
#include <bs/kernel/config.h>
#include <bs/detail/sharded_mutex.h>

#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/pattern_formatter.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>

constexpr auto FILE_LOG_PATTERN    = std::string_view{ "[%Y-%m-%d %T.%e] [%P] [%L] [%*] %v" };
constexpr auto CONSOLE_LOG_PATTERN = std::string_view{ "[%L] %v" };
constexpr auto CUSTOM_LOG_PATTERN  = std::string_view{ "%v" };
constexpr auto LOG_FNAME_PREFIX    = std::string_view{ "bs_" };
constexpr auto CUSTOM_TAG_FIELD    = std::string_view{ "[%*]" };

constexpr auto ROTATING_FSIZE_DEFAULT = 1024*1024*5;
constexpr auto DEF_FLUSH_INTERVAL = 1;
constexpr auto DEF_FLUSH_LEVEL = spdlog::level::err;

using namespace blue_sky::detail;
using namespace blue_sky;
namespace fs = std::filesystem;

NAMESPACE_BEGIN()
///////////////////////////////////////////////////////////////////////////////
//  log-related globals
//
// fallback null sink
const auto& null_sink() {
	static const auto nowhere = std::make_shared<spdlog::sinks::null_sink_mt>();
	return nowhere;
}

const auto& null_logger() {
	// unregistered standalone logger
	static const auto nowhere = std::make_shared<spdlog::logger>("null", null_sink());
	return nowhere;
}

// flag that indicates whether we have switched to mt logs
auto& are_logs_async() {
	static std::atomic<bool> mt_state{false};
	return mt_state;
}

// read value from config or return default value if kernel isn't yet configured
template<typename T>
auto configured_value(std::string_view key, T def_value) {
	return kernel::config::is_configured() ?
		get_or(kernel::config::config(), key, std::move(def_value)) :
		def_value;
}

///////////////////////////////////////////////////////////////////////////////
//  custom pattern flag that adds predefined tag to every record
//
struct custom_tag_flag : public spdlog::custom_flag_formatter {
	static auto tag() -> std::string_view& {
		static auto tag_ = std::string_view{};
		return tag_;
	}

	// [NOTE] lock is aquired only on tag update
	static auto update_tag(std::string t) {
		// tag value storage
		static auto tagv_ = std::string{};
		static auto mut = std::mutex{};
		std::lock_guard _{ mut };
		tagv_ = std::move(t);
		tag() = tagv_;
	}

	auto format(const spdlog::details::log_msg&, const std::tm&, spdlog::memory_buf_t& dest)
	-> void override {
		auto& t = tag();
		dest.append(t.data(), t.data() + t.size());
	}

	auto clone() const -> std::unique_ptr<custom_flag_formatter> override {
		return std::make_unique<custom_tag_flag>();
	}
};

auto make_formatter(std::string pat_format) {
	// remove custom tag field if tag is empty, add (if missing) if tag is non-empty
	const auto pos = pat_format.find(CUSTOM_TAG_FIELD);
	if(custom_tag_flag::tag().empty() && pos != std::string::npos) {
		pat_format.erase(pos, CUSTOM_TAG_FIELD.size());
		// try to remove additional space before/after removed tag
		if(pos < pat_format.size() && std::isspace(pat_format[pos]))
			pat_format.erase(pos, 1);
	}
	// make formatter with custom flag
	auto res = std::make_unique<spdlog::pattern_formatter>();
	res->add_flag<custom_tag_flag>('*').set_pattern(std::move(pat_format));
	return res;
}

/*-----------------------------------------------------------------------------
 *  sinks manager
 *-----------------------------------------------------------------------------*/
struct sinks_registry : public detail::sharded_same_mutex<std::mutex, 3> {
	using sink_ptr = spdlog::sink_ptr;
	using sinks_storage_t = std::multimap<std::string, sink_ptr, std::less<>>;

	// guards for different sinks groups
	enum SinkGroup : std::uint8_t { File = 0, Console = 1, Custom = 2 };
	using guard_t = detail::sharded_same_mutex<std::mutex, 3>;

	static auto group_guard() -> guard_t& {
		static auto guard_ = guard_t{};
		return guard_;
	}

	// access different sinks storages
	template<SinkGroup S>
	static auto group() -> sinks_storage_t& {
		if constexpr(S == File) {
			static auto sinks_ = sinks_storage_t{};
			return sinks_;
		}
		else if constexpr(S == Console) {
			static auto sinks_ = sinks_storage_t{};
			return sinks_;
		}
		else {
			static auto sinks_ = sinks_storage_t{};
			return sinks_;
		}
	}

	template<SinkGroup S>
	static constexpr auto group_name() -> const char* {
		if constexpr(S == File) return "file";
		else if constexpr(S == Console) return "console";
		else return "custom";
	}

	template<SinkGroup S>
	static constexpr auto def_format_pattern() -> std::string_view {
		if constexpr(S == File)
			return FILE_LOG_PATTERN;
		else if constexpr(S == Console)
			return CONSOLE_LOG_PATTERN;
		else
			return CUSTOM_LOG_PATTERN;
	}

	// manipulate with sinks
	template<SinkGroup S, typename F>
	static auto apply(F&& f) {
		auto solo = group_guard().lock<S>();
		auto& sinks = group<S>();
		for_each(sinks.begin(), sinks.end(), std::forward<F>(f));
	}

	template<SinkGroup S, typename F>
	static auto apply(F&& f, std::string_view name) {
		auto solo = group_guard().lock<S>();
		auto& sinks = group<S>();
		const auto impl = [&f](auto r) {
			for(auto i = r.first; i != r.second; ++i)
				f(i->second);
		};
		// apply for exactly matched sinks
		impl(sinks.equal_range(name));
		// and for sinks with empty name
		impl(sinks.equal_range({}));
	}

	template<SinkGroup S>
	static auto add_sink(sink_ptr sink, std::string name) {
		auto solo = group_guard().lock<S>();
		auto& sinks = group<S>();
		refresh_format<S>(sink, name);
		sinks.emplace(std::move(name), std::move(sink));
	}

	// matches sink & optionally name
	template<SinkGroup S>
	static auto remove_sink(const sink_ptr& sink, std::string_view name = {}) -> void {
		auto solo = group_guard().lock<S>();
		auto& sinks = group<S>();
		for(auto s = sinks.begin(); s != sinks.end();) {
			if(s->second == sink && (name.empty() ? true : s->first == name))
				s = sinks.erase(s);
			else ++s;
		}
	}

	template<SinkGroup S>
	static auto clear() -> void {
		auto solo = group_guard().lock<S>();
		group<S>().clear();
	}

	static auto clear_all() -> void {
		clear<File>();
		clear<Console>();
		clear<Custom>();
	}

	template<SinkGroup S, typename F>
	static auto get_or_make(std::string_view name, F&& sink_maker) -> sink_ptr {
		auto solo = group_guard().lock<S>();
		auto& sinks = group<S>();
		if(auto s = sinks.find(name); s!= sinks.end())
			return s->second;
		else {
			try {
				if(auto new_sink = sink_maker()) {
					refresh_format<S>(new_sink, name);
					sinks.emplace(name, new_sink);
					return new_sink;
				}
			}
			catch(...) {}
		}
		return nullptr;
	}

	template<SinkGroup S>
	static auto refresh_format(const sink_ptr& sink, std::string_view name) {
		sink->set_formatter(make_formatter(configured_value(
			std::string("logger.") + std::string{name} + "-" + group_name<S>() + "-format",
			std::string{ def_format_pattern<S>() }
		)));
	}

	template<SinkGroup S>
	static auto refresh_format() {
		static auto impl = [](auto& s) {
			const auto& [name, sink] = s;
			refresh_format<S>(sink, name);
		};
		apply<S>(impl);
	}

	// extract sinks matching given name from all groups
	static auto get(std::string_view name) {
		auto res = std::vector<sink_ptr>{};
		const auto extracter = [&](auto& sink) {
			res.emplace_back(sink);
		};
		apply<File>(extracter, name);
		apply<Console>(extracter, name);
		apply<Custom>(extracter, name);
		return res;
	}
};

/*-----------------------------------------------------------------------------
 *  create sinks
 *-----------------------------------------------------------------------------*/
// purpose of this function is to find log filename that is not locked by another process
// if `logger_name` is set -- tune sink with configured values
auto create_file_sink(const std::string& logger_name) -> spdlog::sink_ptr {
	// extract desired log filename
	const auto def_fname = std::string(LOG_FNAME_PREFIX) + logger_name + ".log";
	const auto desired_fname = configured_value(
		std::string("logger.") + logger_name + "-file-name", def_fname
	);
	// create sink
	auto res = sinks_registry::get_or_make<sinks_registry::File>(logger_name, [&]() -> spdlog::sink_ptr {
		// create intermediate dirs
		const auto logf = fs::path(desired_fname);
		const auto logf_ext = logf.extension();
		const auto logf_parent = logf.parent_path();
		const auto logf_body = logf_parent / logf.stem();
		if(!logf_parent.empty()) {
			std::error_code er;
			if(!fs::exists(logf_parent, er) && !er)
				fs::create_directories(logf_parent, er);
			if(er) {
				std::cerr << "[E] Failed to create/access path of log file " << desired_fname
					<< ": " << er.message() << std::endl;
				return nullptr;
			}
		}

		const auto fsize = configured_value(
			std::string("logger.") + logger_name + "-file-size", ROTATING_FSIZE_DEFAULT
		);
		for(int i = 0; i < 100; i++) {
			auto cur_logf = logf_body;
			if(i)
				cur_logf += std::string("_") + std::to_string(i);
			cur_logf += logf_ext;

			try {
				if(auto res = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(cur_logf.string(), fsize, 1)) {
					std::cout << "[I] Using log file " << cur_logf.string() << std::endl;
					return res;
				}
			} catch(...) {}
		}
		return nullptr;
	});

	// print err if couldn't create log file
	if(!res) {
		std::cerr << "[E] Failed to create log file " << desired_fname << std::endl;
		return null_sink();
	}
	return res;
}

// if `logger_name` is set -- tune sink with configured values
template<typename Sink>
auto create_console_sink(const std::string& logger_name) -> spdlog::sink_ptr {
	auto res = sinks_registry::get_or_make<sinks_registry::Console>(logger_name, [&]() -> spdlog::sink_ptr {
		return std::make_shared<Sink>();
	});
	return res ? res : null_sink();
}

/*-----------------------------------------------------------------------------
 *  create loggers
 *-----------------------------------------------------------------------------*/
template< typename... Sinks >
auto create_logger(const char* logger_name) -> std::shared_ptr<spdlog::logger> {
	auto S = sinks_registry::get(logger_name);
	try {
		auto L = std::make_shared<spdlog::logger>( logger_name, std::begin(S), std::end(S) );
		spdlog::register_logger(L);
		return L;
	}
	catch(...) {}
	return null_logger();
}

template< typename... Sinks >
auto create_async_logger(const char* logger_name) -> std::shared_ptr<spdlog::logger> {
	auto S = sinks_registry::get(logger_name);
	try {
		// create thread pool if not created yet
		if(auto tp = spdlog::thread_pool(); !tp)
			spdlog::init_thread_pool(spdlog::details::default_async_q_size, 1);
		// create non-blocking logger
		auto L = std::make_shared<spdlog::async_logger>(
			logger_name, std::begin(S), std::end(S), spdlog::thread_pool(),
			spdlog::async_overflow_policy::overrun_oldest
		);
		spdlog::register_logger(L);
		// setup flushing policy
		L->flush_on(static_cast<spdlog::level::level_enum>(configured_value(
			std::string("logger.") + logger_name + "-flush-level",
			std::uint8_t(DEF_FLUSH_LEVEL)
		)));
		return L;
	}
	catch(...) {}
	return null_logger();
}

///////////////////////////////////////////////////////////////////////////////
//  builtin 'out' & 'err' loggers
//
struct builtin_loggers {
	static auto& storage() {
		static auto storage_ = std::array<std::optional<log::bs_log>, 2>{
			std::make_optional<log::bs_log>("out"),
			std::make_optional<log::bs_log>("err")
		};
		return storage_;
	}

	static auto& guard() {
		static auto guard_ = std::mutex{};
		return guard_;
	}

	static auto reset() -> void {
		auto solo = std::lock_guard{ guard() };
		storage() = {
			std::make_optional<log::bs_log>("out"),
			std::make_optional<log::bs_log>("err")
		};
	}

	static auto clear() -> void {
		auto solo = std::lock_guard{ guard() };
		auto& nowhere = null_logger();
		storage() = {
			std::make_optional<log::bs_log>(nowhere),
			std::make_optional<log::bs_log>(nowhere)
		};
	}

	template<int i>
	static auto& get() {
		auto solo = std::lock_guard{ guard() };
		return *storage()[i];
	}
};

NAMESPACE_END() // hidden namespace

/*-----------------------------------------------------------------------------
 *  get/create spdlog::logger backend for bs_log
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(log)

auto get_logger(const char* logger_name) -> std::shared_ptr<spdlog::logger> {
	// do spdlog initialization before first logger ever created
	static std::once_flag init_flag_;
	std::call_once(init_flag_, [] {
		// [NOTE] disable auto-registering, beacuse we do it manually
		spdlog::set_automatic_registration(false);
		// set global minimum flush level
		spdlog::flush_on(DEF_FLUSH_LEVEL);
	});

	// if log already registered -- return it
	if(auto L = spdlog::get(logger_name))
		return L;

	// ensure file & console sinks are pre-created
	std::string_view(logger_name) == "err" ?
		create_console_sink<spdlog::sinks::stderr_color_sink_mt>(logger_name) :
		create_console_sink<spdlog::sinks::stdout_color_sink_mt>(logger_name);
	if(kernel::config::is_configured())
		create_file_sink(logger_name);

	// create new logger
	return are_logs_async() ?
		create_async_logger(logger_name) :
		create_logger(logger_name);
}

auto set_custom_tag(std::string tag) -> void {
	custom_tag_flag::update_tag(std::move(tag));
	sinks_registry::refresh_format<sinks_registry::File>();
	sinks_registry::refresh_format<sinks_registry::Console>();
	sinks_registry::refresh_format<sinks_registry::Custom>();
}

NAMESPACE_END(log)

/*-----------------------------------------------------------------------------
 *  kernel logging subsyst impl
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN(kernel::detail)

logging_subsyst::logging_subsyst() {
	// ensure that builtin loggers are created in the very beginning
	[[maybe_unused]] const auto& _ = builtin_loggers::storage();
}

auto logging_subsyst::null_logger() -> const std::shared_ptr<spdlog::logger>& {
	return ::null_logger();
}

// switch between mt- and st- logs
auto logging_subsyst::toggle_async(bool turn_on) -> void {
	if(are_logs_async().exchange(turn_on) != turn_on) {
		// if we're switching from MT -> ST, wait until all pending messages are printed
		// and shutdown thread pool
		if(!turn_on) spdlog::shutdown();
		// reset builtin loggers with new mode
		builtin_loggers::reset();
			// setup periodic flush
		if(turn_on)
			spdlog::flush_every(std::chrono::seconds(configured_value(
				"logger.flush-interval", DEF_FLUSH_INTERVAL
			)));
	}
}

auto logging_subsyst::shutdown() -> void {
	// wait all messages are flushed & stop spdlog threads
	spdlog::shutdown();
	// clear builtin loggers & sinks
	builtin_loggers::clear();
	sinks_registry::clear_all();
}

auto logging_subsyst::add_custom_sink(spdlog::sink_ptr sink, const std::string& logger_name) -> void {
	if(!sink) return;
	// add new sink to regisrty
	sinks_registry::remove_sink<sinks_registry::Custom>(sink, logger_name);
	sinks_registry::add_sink<sinks_registry::Custom>(sink, logger_name);

	if(logger_name.empty())
		// append sink to existing loggers
		spdlog::apply_all([&](auto logger) { logger->sinks().push_back(sink); });
	else if(auto logger = spdlog::get(logger_name))
		logger->sinks().push_back(std::move(sink));
}

auto logging_subsyst::remove_custom_sink(spdlog::sink_ptr sink, const std::string& logger_name) -> void {
	if(!sink) return;
	auto clear_logger = [&](auto logger) {
		auto& Ls = logger->sinks();
		Ls.erase(std::remove(Ls.begin(), Ls.end(), sink), Ls.end());
	};

	if(logger_name.empty())
		spdlog::apply_all(std::move(clear_logger));
	else if(auto logger = spdlog::get(logger_name))
		clear_logger(std::move(logger));

	sinks_registry::remove_sink<sinks_registry::Custom>(sink, logger_name);
}

NAMESPACE_END(kernel::detail)

/*-----------------------------------------------------------------
 * access to main log channels
 *----------------------------------------------------------------*/
log::bs_log& bsout() {
	return builtin_loggers::get<0>();
}

log::bs_log& bserr() {
	return builtin_loggers::get<1>();
}

NAMESPACE_END(blue_sky)
