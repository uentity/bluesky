/// @author uentity
/// @date 19.11.2018
/// @brief Kernel config subsystem implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/defaults.h>
#include <bs/log.h>
#include <bs/kernel/radio.h>

#include "config_subsyst.h"
#include "radio_subsyst.h"
#include "../tree/private_common.h"

#include <caf/config_option_adder.hpp>
#include <caf/parser_state.hpp>
#include <caf/init_global_meta_objects.hpp>
#include <caf/detail/config_consumer.hpp>
#include <caf/detail/parser/read_config.hpp>
#include <caf/detail/parser/read_string.hpp>
#include <caf/io/middleman.hpp>

#include <fmt/ostream.h>

#include <atomic>
#include <sstream>
#include <fstream>
#include <stdlib.h>

NAMESPACE_BEGIN(blue_sky::kernel::detail)
using namespace caf;
namespace fs = std::filesystem;
using string_list = config_subsyst::string_list;

// hidden helper functions
NAMESPACE_BEGIN()

struct config_iter {
	std::istream* conf;
	char ch;

	explicit config_iter(std::istream* istr) : conf(istr) {
		conf->get(ch);
	}

	config_iter() : conf(nullptr), ch('\0') {
		// nop
	}

	config_iter(const config_iter&) = default;

	config_iter& operator=(const config_iter&) = default;

	inline char operator*() const {
		return ch;
	}

	inline config_iter& operator++() {
		conf->get(ch);
		return *this;
	}
};

struct config_sentinel {};

bool operator!=(config_iter iter, config_sentinel) {
	return !iter.conf->fail();
}

auto extract_config_file_path(caf::config_option_set& opts, caf::settings& S, string_list& args)
-> std::string {
	auto ptr = opts.qualified_name_lookup("global.config-file");
	CAF_ASSERT(ptr != nullptr);
	string_list::iterator i;
	string_view path;
	std::tie(i, path) = find_by_long_name(*ptr, args.begin(), args.end());
	// [TODO] add CAF errors processing
	if(i == args.end())
		return "";
	if(path.empty()) {
		args.erase(i);
		return "";
	}
	config_value val{path};
	if(auto err = ptr->sync(val); !err) {
		auto res = caf::get<std::string>(val);
		put(S, "config-file", std::move(val));
		return res;
	}
	return "";
}

struct indentation {
	size_t size;
};

indentation operator+(indentation x, size_t y) noexcept {
	return {x.size + y};
}

std::ostream& operator<<(std::ostream& out, indentation indent) {
	for(size_t i = 0; i < indent.size; ++i)
		out.put(' ');
	return out;
}

void print(const config_value::dictionary& xs, indentation indent, std::ostream& sout) {
	for(const auto& kvp : xs) {
		if(kvp.first == "dump-config")
			continue;
		if(auto submap = get_if<config_value::dictionary>(&kvp.second)) {
			sout << indent << kvp.first << " {\n";
			print(*submap, indent + 2, sout);
			sout << indent << "}\n";
		}
		else if(auto lst = get_if<config_value::list>(&kvp.second)) {
			if(lst->empty()) {
				sout << indent << kvp.first << " = []\n";
			}
			else {
				sout << indent << kvp.first << " = [\n";
				auto list_indent = indent + 2;
				for (auto& x : *lst)
					sout << list_indent << to_string(x) << ",\n";
				sout << indent << "]\n";
			}
		}
		else {
			sout << indent << kvp.first << " = " << to_string(kvp.second) << '\n';
		}
	}
}

// separetely store configured timeouts for faster access
// returns { normal timeout, long op timeout }
using timeouts_pair = std::pair<std::atomic<timespan>, std::atomic<timespan>>;
inline auto timeouts() -> timeouts_pair& {
	static auto ts_ = timeouts_pair{};
	return ts_;
}

auto reset_timeouts(timespan typical, timespan slow) -> void {
	auto& [normal_to, long_to] = timeouts();
	normal_to.store(typical, std::memory_order_relaxed);
	long_to.store(slow, std::memory_order_relaxed);
}

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  kernel_config_subsyst impl
 *-----------------------------------------------------------------------------*/
config_subsyst::config_subsyst() {
	///////////////////////////////////////////////////////////////////////////////
	//  add config options
	//
	using opt_group = config_option_adder;
	opt_group{confopt_, "global"}
		.add<bool>("help,h?", "print help and exit")
		.add<bool>("long-help", "print all help options and exit")
		.add<bool>("dump-config", "print configuration in INI format and exit")
		.add<std::string>("config-file", "BS config file path")
	;
	opt_group{confopt_, "path"}
		.add<std::string>("kernel", "Path to blue-sky kernel library")
		.add<string_list>("plugins", "Paths list to blue-sky plugins")
	;
	opt_group{confopt_, "logger"}
		.add<std::string>("out-file-name", "Path to stdout log file")
		.add<std::string>("err-file-name", "Path to stderr log file")
		.add<std::uint64_t>("out-file-size", "Size of rotating stdout log file")
		.add<std::uint64_t>("err-file-size", "Size of rotating stderr log file")
		.add<std::string>("out-file-format", "Format of stdout log messages")
		.add<std::string>("err-file-format", "Format of stderr log messages")
		.add<std::string>("out-console-format", "Format of stdout log messages in console")
		.add<std::string>("err-console-format", "Format of stdout log messages in console")
		.add<std::uint8_t>("out-flush-level", "Minimum message level that triggers out log flush")
		.add<std::uint8_t>("err-flush-level", "Minimum message level that triggers err log flush")
		.add<std::uint32_t>("flush-interval", "Multithreaded logs background flush interval")
	;
	opt_group(confopt_, "radio")
		.add<std::uint16_t>("port", "Port number for main BS network interface")
		.add<std::uint16_t>("groups-port", "Port number for publishing actor groups")
		.add<timespan>("timeout", "Generic default timeout for actor operations")
		.add<timespan>("long-timeout", "Timeout for long resource-consuming tasks")
		.add<bool>("await_actors_before_shutdown",
			"Do we have to wait until all actors terminate on kernel shutdown?")
	;

	/*-----------------------------------------------------------------------------
	*  Logic here is the following
	*  1. Conf file from the latter path override previous one
	*  2. For UNIX order is the following:
	*  	/etc/bs/bs.conf
	*  	/home/$USER/.bs/bs.conf
	*  3. For Windows order is the following:
	*  	%ALLUSERSPROFILE%\bs\bs.conf (C:\ProgramData\...)
	*  	%APPDATA%\bs\bs.conf (C:\Users\%USER%\AppData\Roaming\...)
	*  4. bs.conf from . dir is the last one
	*-----------------------------------------------------------------------------*/
	// as fallback add possibility to read config from current path
	conf_path_.push_back("bs.conf");
#ifdef UNIX
	conf_path_.emplace_back("/etc/bs/bs.conf");
	conf_path_.push_back( fs::path(::getenv("HOME")) / ".bs/bs.conf" );
#else // WINDOWS
	conf_path_.push_back( fs::path(::getenv("ALLUSERSPROFILE")) / "bs" / "bs.conf" );
	conf_path_.push_back( fs::path(::getenv("USERPROFILE")) / "bs" / "bs.conf" );
#endif // UNIX
}

auto config_subsyst::configure(string_list args, std::string ini_fname, bool force) -> void {
	// build list of config files to parse
	std::vector<fs::path> ini2parse;
	// first read predefined configs
	if(!kernel_configured || force)
		std::copy(conf_path_.begin(), conf_path_.end(), std::back_inserter(ini2parse));
	// then custom configs passed from CLI and `ini_fname` - overrides predefined
	for(auto& p : std::array<std::string, 2>{
		extract_config_file_path(confopt_, confdata_, args), std::move(ini_fname)
	}) {
		if( !p.empty() && std::find(conf_path_.begin(), conf_path_.end(), p) == conf_path_.end() )
			ini2parse.emplace_back(std::move(p));
	};

	// in force mode remove existing content of config data
	if(force) confdata_.clear();

	// read each INI file
	if(!ini2parse.empty()) {
		bsout() << "--------> [configure]" << bs_end;
		bsout() << "Try to load following config files:" << bs_end;
		for(const auto& ini_path : ini2parse) {
			auto ini = std::ifstream(ini_path);
			bool status = false;
			if((status = ini.good())) {
				caf::detail::config_consumer consumer{confopt_, confdata_};
				caf::parser_state<config_iter, config_sentinel> res{config_iter{&ini}};
				caf::detail::parser::read_config(res, consumer);
				if(res.i != res.e) {
					status = false;
					bserr() << log::W("*** error in {} [line {} col {}]: {}")
						<< ini_path.string() << res.line << res.column << to_string(res.code) << log::end;
				}
			}
			bsout() << "{} - {}" << ini_path.string() << (status ? "OK" : "Fail") << log::end;

			// try to read CAF config from the same dir as BS config
			const auto caf_ini_path = ini_path.parent_path() / "caf.conf";
			if(( ini = std::ifstream(caf_ini_path) )) {
				const auto er = actor_cfg_.parse({}, ini);
				bsout() << "CAF: {} - {}" << caf_ini_path.string() << (er ? "Fail" : "OK") << log::end;
				if(er)
					bsout() << "{}" << to_string(er) << log::end;
			}
		}
	}

	// CLI options override the content of the INI file.
	using std::make_move_iterator;
	auto res = confopt_.parse(confdata_, args);
	if (res.second != args.end()) {
		if (res.first != pec::success && starts_with(*res.second, "-")) {
			bserr() << log::W("error: at CLI config argument \"{}\": {}") << *res.second
				<< to_string(res.first) << log::end;
		}
	};

	// Generate help text if needed.
	// These options are one-shot
	if(get_or(confdata_, "help", false) || get_or(confdata_, "long-help", false)) {
		bool long_help = get_or(confdata_, "long-help", false);
		bsout() << confopt_.help_text(!long_help) << log::end;
		put(confdata_, "help", false);
		put(confdata_, "long-help", false);
	}
	// Generate INI dump if needed.
	if(get_or(confdata_, "dump-config", false)) {
		std::stringstream confdump;
		print(confdata_, indentation{0}, confdump);
		bsout() << confdump.str() << log::end;
		put(confdata_, "dump-config", false);
	}

	// update timeout values
	reset_timeouts(
		get_or( confdata_, "radio.timeout", defaults::radio::timeout ),
		get_or( confdata_, "radio.long-timeout", defaults::radio::long_timeout )
	);

	// [NOTE] load networking module after kernel & CAF are configured (do it only once!)
	if(!kernel_configured)
		actor_cfg_.load<caf::io::middleman>();

	kernel_configured = true;
}

auto config_subsyst::clear_confdata() -> void {
	confdata_.clear();
	kernel_configured = false;
}

auto config_subsyst::is_configured() -> bool {
	return kernel_configured;
}

bool config_subsyst::kernel_configured = false;

auto radio_subsyst::reset_timeouts(timespan typical, timespan slow) -> void {
	kernel::detail::reset_timeouts(typical, slow);
}

NAMESPACE_END(blue_sky::kernel::detail)

NAMESPACE_BEGIN(blue_sky::kernel::radio)

auto timeout(bool for_long_task) -> timespan {
	auto& [normal_to, long_to] = kernel::detail::timeouts();
	return for_long_task ?
		long_to.load(std::memory_order_relaxed) :
		normal_to.load(std::memory_order_relaxed);
}

NAMESPACE_END(blue_sky::kernel::radio)
