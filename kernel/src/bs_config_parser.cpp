#include "bs_config_parser.h"
#include "bs_report.h"
#include "bs_exception.h"
#include "bs_tree.h"
#include "bs_misc.h"

#include <boost/regex.hpp>

#include <iostream>
#include <sstream>
#include <fstream>

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
	// Service functions
	string trim_whitespace(string &str);
	std::string trim_icommas (std::string &str);
	bs_cfg_p::vstr_t split_path_list (const string &str);
	void req_split_path (bs_cfg_p::vstr_t &path);
	void get_leaf_win(string &container_, const string &src);

	string trim_whitespace(string &str) {
		std::ostringstream t(std::ios::out | std::ios::binary);
		std::ostream_iterator<char, char> oi(t);
		regex expr("^[ \t]+|[ \t]+$");
		regex_replace(oi, str.begin(), str.end(),
      expr, "", boost::match_default | boost::format_all);
		return t.str();
	}

	std::string trim_icommas (std::string &str) {
		std::ostringstream t(std::ios::out | std::ios::binary);
		std::ostream_iterator<char, char> oi(t);
		regex expr("^[\"]+|[\"]+$");
		regex_replace(oi, str.begin(), str.end(),
      expr, "", boost::match_default | boost::format_all);
		return t.str();
	}

	bs_cfg_p::vstr_t split_path_list (const string &str) {
		bs_cfg_p::vstr_t path;

		path.push_back(str);
		req_split_path(path);
		for (size_t i = 0; i < path.size(); ++i)
			path[i] = trim_icommas(path[i]);
		return path;
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

	void get_leaf_win(string &container_, const string &src)
	{
		for (size_t i = src.size() - 1; i < src.size(); --i) {
			if (src[i] == '\\')
				break;
			container_ = src[i] + container_;
		}
	}

	// class methods
	bs_cfg_p::bs_cfg_p () {
		
	}

	void bs_cfg_p::parse_strings(const string &str, bool append) {
		string key;

		vector<string> res;
		string tmp (str);
		regex expression("^([^\?#]*)[=](.*)$");
		regex_split(back_inserter(res),tmp,expression,match_default);
		if(res.size() > 1) {
			
			vstr_t tval = split_path_list(trim_whitespace(res[1]));
			string tstr = trim_whitespace(res[0]);
			if (append) {
				vstr_t new_val = env_mp[tstr];
				new_val.insert(new_val.end(),tval.begin(),tval.end());
				env_mp[tstr] = new_val;
			}
			else // CHANGE_ENVS
				env_mp[tstr] = tval;
		}
	}

	void bs_cfg_p::read_file (const char *filename) {
		std::ifstream srcfile(filename);

		if (!srcfile) {
			BSERROR << "Main Blue-Sky Config Parser: No config file \"" << filename << "\"" << bs_end;
			return;
		}

		for (;!srcfile.eof();) {
			string str;
			getline(srcfile,str,'\n');
			parse_strings (str);
		}
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

	struct vars_t {
		string config [2];

		vars_t () {
#ifdef UNIX
			char *userpath;
			userpath = getenv("HOME");

			config[0] = "/etc/blue-sky/blue-sky.conf";
			config[1] = string(userpath) + string("/.blue-sky/blue-sky.conf");
#else // !UNIX
			string userappdata = getenv("APPDATA");
			string appdata;

			get_leaf_win(appdata,userappdata);
			appdata = string(getenv("ALLUSERSPROFILE")) + string("\\") + appdata;

			config[0] = appdata + "\\blue-sky\\blue-sky.conf";
			config[1] = userappdata + "\\blue-sky\\blue-sky.conf";
#endif // UNIX
		}
	};

	struct wcfg {

		bs_cfg_p cfg_;
		vars_t vars;

		bs_cfg_p& (wcfg::*ref_fun_)();

		wcfg()
			: ref_fun_(&wcfg::initial_cfg_getter)
		{}

		void init_cfg () {
			cfg_.read_file(vars.config[0].c_str());
			cfg_.read_file(vars.config[1].c_str());

			/*if (cfg_.env_mp["BLUE_SKY_ARCH"].size ()) {
				if (cfg_.env_mp["BLUE_SKY_ARCH"][0] == string("32"))
					cfg_.env_mp["BLUE_SKY_ARCH"][0] = "";
			}
			else
			cfg_.env_mp["BLUE_SKY_ARCH"].resize(1);*/

			if (!cfg_.env_mp["BLUE_SKY_PATH"].size ()) {
				BSERROR << "BLUE_SKY_PATH variable not found in your configs... Setting it to \"./\"" << bs_end;
				cfg_.env_mp["BLUE_SKY_PATH"].resize(1);
				cfg_.env_mp["BLUE_SKY_PATH"][0] = "./";
			}

			if (!cfg_.env_mp["BLUE_SKY_PREFIX"].size ()) {
				BSERROR << "BLUE_SKY_PREFIX variable not found in your configs... Setting it to \"./\"" << bs_end;
				cfg_.env_mp["BLUE_SKY_PREFIX"].resize(1);
				cfg_.env_mp["BLUE_SKY_PREFIX"][0] = "./";
			}

			//cfg_.env_mp["BLUE_SKY_PATH"].insert(cfg_.env_mp["BLUE_SKY_PATH"].begin(),cfg_.env_mp["BLUE_SKY_PREFIX"][0]
			//+ string("/lib") + cfg_.env_mp["BLUE_SKY_ARCH"][0]);
#ifdef UNIX
			cfg_.env_mp["BLUE_SKY_PLUGINS_PATH"].insert(cfg_.env_mp["BLUE_SKY_PLUGINS_PATH"].begin(),string(getenv("HOME")) + string("/.blue-sky/plugins"));
			cfg_.env_mp["BLUE_SKY_PLUGINS_PATH"].insert(cfg_.env_mp["BLUE_SKY_PLUGINS_PATH"].begin(),cfg_.env_mp["BLUE_SKY_PREFIX"][0] + string("/share/blue-sky/plugins"));
#else // WINDOWS
			cfg_.env_mp["BLUE_SKY_PLUGINS_PATH"].insert(cfg_.env_mp["BLUE_SKY_PLUGINS_PATH"].begin(),string(getenv("APPDATA")) + string("\\blue-sky\\plugins"));
			cfg_.env_mp["BLUE_SKY_PLUGINS_PATH"].insert(cfg_.env_mp["BLUE_SKY_PLUGINS_PATH"].begin(),cfg_.env_mp["BLUE_SKY_PREFIX"][0] + string("\\plugins"));
#endif // UNIX
			
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

	typedef SingletonHolder< wcfg, CreateUsingNew,
		FollowIntoDeath::With< DefaultLifetime >::AsMasterLifetime > cfg_holder;

	template< >
	BS_API bs_cfg_p& singleton< bs_cfg_p >::Instance()
	{
		return cfg_holder::Instance().cfg_ref();
	}

	BS_API bs_cfg_p::vstr_t
	bs_config::operator [] (const char *e) {
		return cfg::Instance ()[e];
	}

};
