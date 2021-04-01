/// @file
/// @author uentity
/// @date 02.12.2018
/// @brief Implement BS plugins discover in configured paths
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include <bs/detail/str_utils.h>
#include <bs/kernel/config.h>
#include "plugins_subsyst.h"

#include <deque>
#include <filesystem>
#include <boost/regex.hpp>

#ifdef UNIX
#	define SPL ":"
#else // UNIX
#	define SPL ";"
#endif // UNIX

#define BS_PLUG_PATH "BLUE_SKY_PLUGINS_PATH"
#define STRCAT(a,b) (std::string(a) + b)

namespace fs = std::filesystem;

NAMESPACE_BEGIN(blue_sky::kernel::detail)

using namespace boost;
using vstr_t = std::deque< std::string >;
using fname_set = plugins_subsyst::fname_set;
using string_list = std::vector<std::string>;

/*-----------------------------------------------------------------------------
 *  helpers
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()

std::string trim_icommas(const std::string& str) {
	std::ostringstream t(std::ios::out | std::ios::binary);
	std::ostream_iterator<char, char> oi(t);
	regex expr("^[\"]+|[\"]+$");
	regex_replace(
		oi, str.begin(), str.end(),
		expr, "", boost::match_default | boost::format_all
	);
	return t.str();
}

// recursively split first element into parts
void req_split_path(vstr_t& path) {
	if (path.size() == 0)
		return;

	vstr_t tpath;
	std::string tmp = std::string(path.front());
	regex expression(STRCAT(STRCAT("^(.*)[", SPL), "](.*)$"));
	regex_split(std::back_inserter(tpath), tmp, expression, match_default);
	if(tpath.size() != 0) {
		tpath.insert(tpath.end(), path.begin()+1, path.end());
		req_split_path(tpath);
		path.clear();
		path.assign(tpath.begin(), tpath.end());
	}
}

vstr_t split_path_list(const std::string& str) {
	vstr_t paths;

	paths.push_back(str);
	req_split_path(paths);
	for(auto& path : paths) {
		path = trim_icommas(path);
	}
	return paths;
}

template< typename dest_t, typename value_t >
void push_unique(dest_t& dest, value_t&& value, bool push_front = false) {
	// skip repeating values
	if(std::find(dest.begin(), dest.end(), std::forward< value_t >(value)) != dest.end())
		return;
	// insert value
	if(push_front)
		dest.emplace_front(std::forward< value_t >(value));
	else
		dest.emplace_back(std::forward< value_t >(value));
}

// [NOTE] grows res inplace
void search_files(const char* mask, const fs::path& dir, fname_set& res) {
	try {
		// first file of directory as iterator
		for(const fs::directory_entry& node : fs::directory_iterator(dir)) {
			if(fs::is_directory(node)) continue;
			auto node_path = node.path().string();
			if(regex_search(node_path, regex(std::string("^(.*)") + mask + '$'))) {
				res.emplace(std::move(node_path));
			}
		}
	}
	catch(const fs::filesystem_error&) {
		// [NOTE] don't spam logs with non-found paths, etc
		//BSERROR << log::W(e.what()) << bs_end;
	}
}

NAMESPACE_END() // eof hidden namespace

/*-----------------------------------------------------------------------------
 * `discover_plugins()` impl
 *-----------------------------------------------------------------------------*/
// return set of discovered plugins
auto plugins_subsyst::discover_plugins() -> fname_set {
	// resulting paths container
	std::deque<fs::path> plugins_paths;

	// 1. Extract list of plugin paths from configs
	auto conf_paths = caf::get_or<string_list>(config::config(), "path.plugins", string_list{});
	for(auto& P : conf_paths)
		push_unique(plugins_paths, std::move(P));

	// 2. Check path set via environment
	// env vars have highest priority, insert 'em to beginning
	if(auto pprefix = ::getenv(BS_PLUG_PATH)) {
		auto env_paths = split_path_list(trim(pprefix));
		for(auto& P : env_paths)
			push_unique(plugins_paths, std::move(P));
	}

	// 3. Add some predefined paths
#ifdef UNIX
	push_unique(plugins_paths, fs::path(::getenv("HOME")) / ".blue-sky/plugins");
	push_unique(plugins_paths, "/usr/share/blue-sky/plugins");
#else // WINDOWS
	push_unique(plugins_paths, fs::path(::getenv("ALLUSERSPROFILE")) / "Application Data" / "blue-sky" / "plugins");
	push_unique(plugins_paths, fs::path(::getenv("APPDATA")) / "blue-sky" / "plugins");
#endif // UNIX
	// [TODO] discover path to kernel library as fallback
	// if no paths were set in config files, add current dir as lowest priority search path
	if(plugins_paths.empty()) push_unique(plugins_paths, ".");

	// print some info
	BSOUT << "--------> [discover plugins]" << bs_end;
	BSOUT << "Search for plugins in following discovered paths:" << bs_end;
	for(const auto& path : plugins_paths) {
		BSOUT << path.string() << bs_end;
	}

	// search for plugins inside discovered directories
	fname_set plugins;
	for(const auto& plug_path : plugins_paths) {
#ifdef _WIN32
		search_files(".dll", plug_path, plugins);
#else
		search_files(".so", plug_path, plugins);
#endif
	}

	if(!plugins.size())
		BSOUT << "No plugins were found!" << bs_end;
	else {
		BSOUT << "[Found plugins]" << bs_end;
		for(const auto& plugin : plugins)
			BSOUT << "{}" << plugin << bs_end;
	}

	return plugins;
}

NAMESPACE_END(blue_sky::kernel::detail)
