/// @file
/// @author uentity
/// @date 04.09.2016
/// @brief Routines to find BlueSKy plugins filenames
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "kernel_plugins_subsyst.h"
#include <bs/log.h>
#include <bs/detail/str_utils.h>

#include <deque>
#include <map>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>

#include <sstream>
#include <fstream>
#include <stdlib.h>

#ifdef UNIX
# define SPL ":"
#else // UNIX
# define SPL ";"
#endif // UNIX

#define BS_PATH "BLUE_SKY_PATH"
#define BS_PLUG_PATH "BLUE_SKY_PLUGINS_PATH"
#define STRCAT(a,b) std::string(std::string(a) + std::string(b))

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(detail)

// hidden implementation
namespace {
using namespace boost;

using vstr_t = std::deque< std::string >;
using fname_set = std::set< std::string >;

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
	//for (size_t i = 0; i < path.size(); ++i)
	//	path[i] = trim_icommas(path[i]);
	return paths;
}

static std::string getenv_def(const char* var_name, const std::string& def_val = ".") {
	std::string val = ::getenv(var_name);
	if(val.size())
		return val;
	else return def_val;
}

struct plugins_discover {

	void parse_conf_entry(const std::string& str, bool append) {
		std::string key;

		std::vector<std::string> res;
		std::string tmp (str);
		regex expression("^([^\?#]*)[=](.*)$");
		regex_split(back_inserter(res), tmp, expression, match_default);
		if(res.size() > 1) {
			vstr_t tval = split_path_list(trim(res[1]));
			std::string tstr = trim(res[0]);
			if (append) {
				vstr_t& new_val = bs_path_[tstr];
				new_val.insert(new_val.end(), tval.begin(), tval.end());
			}
			else // CHANGE_ENVS
				bs_path_[tstr] = tval;
		}
	}

	bool read_conf_file(const char *filename, bool append = false) {
		std::ifstream srcfile(filename);

		if(!srcfile) {
			//BSERROR << "Main Blue-Sky Config Parser: No config file \"" << filename << "\"" << bs_end;
			return false;
		}

		std::string buf;
		while(srcfile) {
			std::getline(srcfile, buf);
			parse_conf_entry(buf, append);
		}
		return true;
	}

	template< typename key_t, typename paths_t >
	void add_paths(key_t&& key, const paths_t& what, bool push_front = false, bool replace = false) {
		vstr_t& dest = bs_path_[std::forward< key_t >(key)];
		if(replace)
			dest.clear();

		for(const auto& path : what) {
			add_path(dest, path, push_front);
		}
	}

	template< typename value_t >
	void add_path(vstr_t& dest, value_t&& value, bool push_front = false) {
		//vstr_t& dest = bs_path_[std::forward< key_t >(key)];
		// skip repeating paths
		if(std::find(dest.begin(), dest.end(), std::forward< value_t >(value)) != dest.end())
			return;
		// insert value
		if(push_front)
			dest.emplace_front(std::forward< value_t >(value));
		else
			dest.emplace_back(std::forward< value_t >(value));
	}

	void init_conf_path() {
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
		std::string home_path;
#ifdef UNIX
		conf_path_.push_back("/etc/blue-sky/blue-sky.conf");
		home_path = ::getenv("HOME");
		if(!home_path.empty())
			conf_path_.push_back(home_path + "/.blue-sky/blue-sky.conf");
#else // WINDOWS
		home_path = ::getenv("ALLUSERSPROFILE");
		if(!home_path.empty())
			conf_path_.push_back(home_path + "\\blue-sky\\blue-sky.conf");
		home_path = ::getenv("USERPROFILE");
		if(!home_path.empty())
			conf_path_.push_back(home_path + "\\blue-sky\\blue-sky.conf");
#endif // UNIX
		// as fallback add possibility to read conf_path_ from current path
		conf_path_.push_back("blue-sky.conf");
	}

	void init_plugins_path() {
		// 1. Read paths from config file
		BSOUT << "--------" << bs_end;
		BSOUT << "Try to load following config files:" << bs_end;
		for(const auto& path : conf_path_) {
			BSOUT << "{} - {}" << path
				<< (read_conf_file(path.c_str()) ? "OK" : "Fail") << bs_end;
		}

		// 2. Check if corresponding variables are set via environment
		// env vars have highest priority
		static const char* env_vars[] = { BS_PLUG_PATH, BS_PATH };
		const char* pprefix;
		for(auto var : env_vars) {
			pprefix = ::getenv(var);
			if(pprefix)
				add_paths(var, split_path_list(trim(pprefix)), true);
		}


		// 3. Add some predefined paths as a fallback - lowest priority
		vstr_t& plugins_paths = bs_path_[BS_PLUG_PATH];
		// TODO: discover path to kernel library as fallback
		// if no paths were set in config files, add current dir as highest priority search path
		if(plugins_paths.empty()) add_path(plugins_paths, ".");
#ifdef UNIX
		add_path(plugins_paths, getenv_def("HOME") + "/.blue-sky/plugins", true);
		add_path(plugins_paths, "/usr/share/blue-sky/plugins", true);
#else // WINDOWS
		add_path(plugins_paths, getenv_def("ALLUSERSPROFILE") + "\\Application Data\\blue-sky\\plugins", true);
		add_path(plugins_paths, getenv_def("APPDATA") + "\\blue-sky\\plugins", true);
#endif // UNIX

		// now we literally need to reverse plugins paths order
		// because LoadPlugins will start loading from FIRST path in list
		// and skip all duplicates from tail paths
		std::reverse(plugins_paths.begin(), plugins_paths.end());

		// print some info
		BSOUT << "--------" << bs_end;
		BSOUT << "Search for plugins in discovered paths from BLUE_SKY_PLUGINS_PATH:" << bs_end;
		for(const auto& path : plugins_paths) {
			BSOUT << path << bs_end;
		}
	}

	// NOTE: grows res array inplace
	void search_files(const char* mask, const std::string& dir, fname_set& res) {
		using namespace boost::filesystem;
		//if(dir == NULL) dir = "./";

		try {
#ifndef UNIX
			path pdir(dir, filesystem::native);
#else
			path pdir(dir);
#endif

			// first file of directory as iterator
			for(directory_entry& node : directory_iterator(pdir)) {
				if(is_directory(node))
					continue;
				if(regex_search(node.path().string(), regex(std::string("^(.*)") + mask + '$'))) {
				//if(compare(node.path().string(), mask, "^(.*)", "$"))
					res.emplace(node.path().string());
				}
			}
		}
		catch(const filesystem::filesystem_error &e) {
			BSERROR << log::W(e.what()) << bs_end;
		}
	}

	// main function
	fname_set go() {
		// read config files and init plugins search paths
		init_conf_path();
		init_plugins_path();

		fname_set plugins;
		for(const auto& plug_path : bs_path_[BS_PLUG_PATH]) {
#ifdef _WIN32
			search_files(".dll", plug_path, plugins);
			search_files(".pyd", plug_path, plugins);
#else
			search_files(".so", plug_path, plugins);
#endif
		}
		return plugins;
	}

	std::vector< std::string > conf_path_;
	std::map< std::string, vstr_t > bs_path_;
};

} //eof hidden namespace

NAMESPACE_END(detail)
NAMESPACE_END(blue_sky)

