/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "bs_config_parser.h"
#include "bs_report.h"
#include "bs_exception.h"
#include "bs_tree.h"
#include "bs_misc.h"
#include "bs_kernel.h"

#include <boost/regex.hpp>

#include <sstream>
#include <fstream>
#include <stdlib.h>

#include "loki/Singleton.h"

using namespace std;
using namespace boost;
using namespace Loki;

#ifdef UNIX
# define SPL ":"
#else // UNIX
# define SPL ";"
#endif // UNIX

#define ARCH
#define ARCHCAT(a) string (string(a) + string( ARCH ))
#define BSCAT(a,b) string(string(a) + string(b))

namespace blue_sky {

// Service functions in hidden namespace
namespace {

string trim_whitespace(const string& str) {
	std::ostringstream t(std::ios::out | std::ios::binary);
	std::ostream_iterator<char, char> oi(t);
	regex expr("^[ \t]+|[ \t]+$");
	regex_replace(oi, str.begin(), str.end(),
	expr, "", boost::match_default | boost::format_all);
	return t.str();
}

std::string trim_icommas(const std::string& str) {
	std::ostringstream t(std::ios::out | std::ios::binary);
	std::ostream_iterator<char, char> oi(t);
	regex expr("^[\"]+|[\"]+$");
	regex_replace(oi, str.begin(), str.end(),
	expr, "", boost::match_default | boost::format_all);
	return t.str();
}

void req_split_path (bs_cfg_p::vstr_t &path) {
	if (path.size() == 0)
		return;

	bs_cfg_p::vstr_t tpath;
	string tmp = string(path.front());
	regex expression(BSCAT(BSCAT("^(.*)[",SPL),"](.*)$"));
	regex_split(back_inserter(tpath),tmp,expression,match_default);
	if(tpath.size() != 0) {
		tpath.insert(tpath.end(),path.begin()+1,path.end());
		req_split_path(tpath);
		path.clear();
		path.assign(tpath.begin(),tpath.end());
	}
}

bs_cfg_p::vstr_t split_path_list(const string &str) {
	bs_cfg_p::vstr_t path;

	path.push_back(str);
	req_split_path(path);
	for (size_t i = 0; i < path.size(); ++i)
		path[i] = trim_icommas(path[i]);
	return path;
}

void get_leaf_win(string &container_, const string &src)
{
	for (size_t i = src.size() - 1; i < src.size(); --i) {
		if (src[i] == '\\')
			break;
		container_ = src[i] + container_;
	}
}

} // eof hidden namespace

// bs_cfg_p implementation

// class methods
bs_cfg_p::bs_cfg_p () {
}

void bs_cfg_p::parse_strings(const string &str, bool append) {
	string key;

	vector<string> res;
	string tmp (str);
	regex expression("^([^\?#]*)[=](.*)$");
	regex_split(back_inserter(res), tmp, expression, match_default);
	if(res.size() > 1) {
		vstr_t tval = split_path_list(trim_whitespace(res[1]));
		string tstr = trim_whitespace(res[0]);
		if (append) {
			vstr_t& new_val = env_mp[tstr];
			new_val.insert(new_val.end(), tval.begin(), tval.end());
		}
		else // CHANGE_ENVS
			env_mp[tstr] = tval;
	}
}

bool bs_cfg_p::read_file (const char *filename) {
	std::ifstream srcfile(filename);

	if(!srcfile) {
		//BSERROR << "Main Blue-Sky Config Parser: No config file \"" << filename << "\"" << bs_end;
		return false;
	}

	while(!srcfile.eof()) {
		string str;
		getline(srcfile, str, '\n');
		parse_strings(str, true);
	}
	return true;
}

void bs_cfg_p::clear_env_map () {
	env_mp.clear();
}

const bs_cfg_p::map_t &bs_cfg_p::env() const {
	return env_mp;
}

bs_cfg_p::vstr_t bs_cfg_p::getenv(const char *e) {
	return env_mp[string(e)];
}

// SINGLETON IMPL
struct conf_path {
	std::vector< std::string > config;

	conf_path() {
		std::string home_path;
#ifdef UNIX
		config.push_back("/etc/blue-sky/blue-sky.conf");
		home_path = ::getenv("HOME");
		if(!home_path.empty())
			config.push_back(home_path + "/.blue-sky/blue-sky.conf");
#else // WINDOWS
		home_path = ::getenv("APPDATA");
		if(!home_path.empty())
			config.push_back(home_path + "\\blue-sky\\blue-sky.conf");
		home_path = ::getenv("ALLUSERSPROFILE");
		if(!home_path.empty())
			config.push_back(home_path + "\\blue-sky\\blue-sky.conf");
#endif // UNIX
		// as fallback add possibility to read config from current path
		config.push_back("blue-sky.conf");
	}
};

struct wcfg {

	bs_cfg_p cfg_;
	conf_path conf_path_;

	bs_cfg_p& (wcfg::*ref_fun_)();

	wcfg()
		: ref_fun_(&wcfg::initial_cfg_getter)
	{}

	static std::string getenv_str(const char* var_name, const char* def_val = ".") {
		const char* val = ::getenv(var_name);
		if(val)
			return val;
		else return def_val;
	}

	template< class Iterator >
	void add_paths(const char* key, const Iterator& from, const Iterator& to) {
		bs_cfg_p::vstr_t& dest = cfg_.env_mp[key];
		dest.insert(dest.end(), from, to);
	}

	void add_paths(const char* key, const bs_cfg_p::vstr_t& what, bool append = true) {
		if(append)
			add_paths(key, what.begin(), what.end());
		else
			cfg_.env_mp[key] = what;
	}

	void init_cfg () {
		const char* pprefix;

		// 1. Read paths from config file
		BSOUT << "--------" << bs_end;
		BSOUT << "Try load config file from following paths:" << bs_end;
		for(ulong i = 0; i < conf_path_.config.size(); ++i) {
			BSOUT << conf_path_.config[i] << " - " <<
				(cfg_.read_file(conf_path_.config[i].c_str()) ? "OK" : "Fail") << bs_end;
		}

		// 2. Check if corresponding variables are set via environment
		// env vars have highest priority
		static const char* env_vars[] = {
			"BLUE_SKY_PLUGINS_PATH", "BLUE_SKY_PATH", "BLUE_SKY_PREFIX"
		};
		for(uint i = 0; i < 3; ++i) {
			pprefix = ::getenv(env_vars[i]);
			if(pprefix)
				add_paths(env_vars[i], split_path_list(trim_whitespace(pprefix)));
		}

		// PREFIX and PATH defaults to current dir
		if(cfg_.env_mp["BLUE_SKY_PATH"].empty())
			cfg_.env_mp["BLUE_SKY_PATH"].push_back(".");
		if(cfg_.env_mp["BLUE_SKY_PREFIX"].empty())
			cfg_.env_mp["BLUE_SKY_PREFIX"].push_back(".");

		// 3. Add some predefined paths as a fallback - lowest priority
		bs_cfg_p::vstr_t& plugins_paths = cfg_.env_mp["BLUE_SKY_PLUGINS_PATH"];
#ifdef UNIX
		plugins_paths.push_front(getenv_str("HOME") + "/.blue-sky/plugins");
		plugins_paths.push_front("/usr/share/blue-sky/plugins");
#else // WINDOWS
		plugins_paths.push_front(getenv_str("ALLUSERSPROFILE") + "\\Application Data\\blue-sky\\plugins");
		plugins_paths.push_front(getenv_str("APPDATA") + "\\blue-sky\\plugins");
#endif // UNIX

		// print some info
		BSOUT << "--------" << bs_end;
		BSOUT << "Search for plugins in discovered paths from BLUE_SKY_PLUGINS_PATH:" << bs_end;
		for(ulong i = 0; i < plugins_paths.size(); ++i) {
			BSOUT << plugins_paths[i] << bs_end;
		}
	}

	bs_cfg_p& usual_cfg_getter() {
		return cfg_;
	}

	bs_cfg_p& initial_cfg_getter() {
		ref_fun_ = &wcfg::usual_cfg_getter;
		init_cfg ();

		return cfg_;
	}

	bs_cfg_p& cfg_ref() {
		return (this->*ref_fun_)();
	}
};

// Singleton instantiation
typedef SingletonHolder< wcfg, CreateUsingNew,
	FollowIntoDeath::With< DefaultLifetime >::AsMasterLifetime > cfg_holder;

template< >
BS_API bs_cfg_p& singleton< bs_cfg_p >::Instance()
{
	return cfg_holder::Instance().cfg_ref();
}

bs_cfg_p::vstr_t
bs_config::operator [] (const char *e) {
	return cfg::Instance ()[e];
}

} // eof blue_sky namespace
