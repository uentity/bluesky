/// @file
/// @author uentity
/// @date 19.11.2018
/// @brief Kernel config subsystem implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include "config_subsyst.h"

#include <caf/config_option_adder.hpp>
#include <caf/detail/ini_consumer.hpp>
#include <caf/detail/parser/read_ini.hpp>
#include <caf/detail/parser/read_string.hpp>
#include <caf/io/middleman.hpp>
#include <fmt/ostream.h>

#include <sstream>
#include <fstream>
#include <stdlib.h>

NAMESPACE_BEGIN(blue_sky::kernel::detail)
using namespace caf;
namespace fs = std::filesystem;
using string_list = config_subsyst::string_list;

// hidden helper functions
NAMESPACE_BEGIN()

struct ini_iter {
	std::istream* ini;
	char ch;

	explicit ini_iter(std::istream* istr) : ini(istr) {
		ini->get(ch);
	}

	ini_iter() : ini(nullptr), ch('\0') {
		// nop
	}

	ini_iter(const ini_iter&) = default;

	ini_iter& operator=(const ini_iter&) = default;

	inline char operator*() const {
		return ch;
	}

	inline ini_iter& operator++() {
		ini->get(ch);
		return *this;
	}
};

struct ini_sentinel { };

bool operator!=(ini_iter iter, ini_sentinel) {
	return !iter.ini->fail();
}

auto extract_config_file_path(string_list& args) -> std::string {
	static constexpr const char needle[] = "--bs-config-file=";
	auto last = args.end();
	auto i = std::find_if(args.begin(), last, [](const std::string& arg) {
			return arg.compare(0, sizeof(needle) - 1, needle) == 0;
			});
	if (i == last)
		return "";
	auto arg_begin = i->begin() + sizeof(needle);
	auto arg_end = i->end();
	if (arg_begin == arg_end) {
		// Missing value.
		// TODO: print warning?
		return "";
	}

	std::string new_conf;
	if (*arg_begin == '"') {
		caf::detail::parser::state<std::string::const_iterator> res;
		res.i = arg_begin;
		res.e = arg_end;
		struct consumer {
			std::string result;
			void value(std::string&& x) {
				result = std::move(x);
			}
		};
		consumer f;
		caf::detail::parser::read_string(res, f);
		if (res.code == pec::success)
			new_conf = std::move(f.result);
		// TODO: else print warning?
	} else {
		// We support unescaped strings for convenience on the CLI.
		new_conf = std::string(arg_begin, arg_end);
	}

	args.erase(i);
	return new_conf;
}

NAMESPACE_END() // eof hidden ns

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

	/*-----------------------------------------------------------------------------
	*  Logic here is the following
	*  1. Conf file from the latter path override previous one
	*  2. For UNIX order is the following:
	*  	/etc/blue-sky/blue-sky.conf
	*  	/home/$USER/.blue-sky/blue-sky.conf
	*  3. For Windows order is the following:
	*  	%ALLUSERSPROFILE%\blue-sky\blue-sky.conf (C:\ProgramData\...)
	*  	%APPDATA%\blue-sky\blue-sky.conf (C:\Users\%USER%\AppData\Roaming\...)
	*  4. blue-sky.conf from . dir is the last one
	*-----------------------------------------------------------------------------*/
	// as fallback add possibility to read config from current path
	conf_path_.push_back("blue-sky.ini");
#ifdef UNIX
	conf_path_.emplace_back("/etc/blue-sky/blue-sky.ini");
	conf_path_.push_back( fs::path(::getenv("HOME")) / ".blue-sky/blue-sky.ini" );
#else // WINDOWS
	conf_path_.push_back( fs::path(::getenv("ALLUSERSPROFILE")) / "blue-sky" / "blue-sky.ini" );
	conf_path_.push_back( fs::path(::getenv("USERPROFILE")) / "blue-sky" / "blue-sky.ini" );
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
		extract_config_file_path(args), std::move(ini_fname)
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
			if ((status = ini.good())) {
				caf::detail::ini_consumer consumer{confopt_, confdata_};
				caf::detail::parser::state<ini_iter, ini_sentinel> res;
				res.i = ini_iter{&ini};
				caf::detail::parser::read_ini(res, consumer);
				if (res.i != res.e) {
					status = false;
					bserr() << log::W("*** error in {} [line {} col {}]: {}")
						<< ini_path.string() << res.line << res.column << to_string(res.code) << log::end;
				}
			}
			bsout() << "{} - {}" << ini_path.string() << (status ? "OK" : "Fail") << log::end;
			// try to read CAF config from the same dir as BS config
			const auto caf_ini_path = ini_path.parent_path() / "caf.ini";
			if(( ini = std::ifstream(caf_ini_path) )) {
				actor_cfg_.parse({}, ini);
				bsout() << "{} - {}" << caf_ini_path.string() << "CAF" << log::end;
			}
		}
	}

	// CLI options override the content of the INI file.
	using std::make_move_iterator;
	auto res = confopt_.parse(confdata_, args);
	if (res.second != args.end()) {
		if (res.first != pec::success) {
			bserr() << log::W("error: at CLI config argument \"{}\": {}") << *res.second
				<< to_string(res.first) << log::end;
		}
	};
	// Generate help text if needed.
	// These options are one-shot
	if(get_or(confdata_, "global.help", false) || get_or(confdata_, "global.long-help", false)) {
		bool long_help = get_or(confdata_, "global.long-help", false);
		bsout() << confopt_.help_text(!long_help) << log::end;
		put(confdata_, "global.help", false);
		put(confdata_, "global.long-help", false);
	}
	// Generate INI dump if needed.
	if(get_or(confdata_, "global.dump-config", false)) {
		std::stringstream confdump;
		confdump << '\n';
		for (auto& category : confdata_) {
			confdump << "[" << category.first << "]\n";
			for (auto& kvp : category.second)
				if (kvp.first != "dump-config")
					confdump << kvp.first << '=' << to_string(kvp.second) << '\n';
		}
		bsout() << confdump.str() << log::end;
		put(confdata_, "global.dump-config", false);
	}

	// load middleman module only once
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

NAMESPACE_END(blue_sky::kernel::detail)

