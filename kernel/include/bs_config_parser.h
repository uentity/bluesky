#ifndef BS_CONFIG_PARSER_H
#define BS_CONFIG_PARSER_H

#include "bs_common.h"

#include <map>
#include <string>
#include <vector>

namespace blue_sky {
struct wcfg;

class BS_API bs_cfg_p {
public:
	friend struct wcfg;

	typedef std::vector<std::string>     vstr_t;
	typedef std::map<std::string,vstr_t> map_t;


	void parse_strings (const std::string &str, bool append = false);
	void read_file (const char *filename);
	void clear_env_map ();
	const map_t &env() const;
	vstr_t getenv(const char *e);

	vstr_t operator[] (const char *e)
	{
		return getenv (e);
	}

private:
	bs_cfg_p ();

	map_t env_mp;
};

typedef singleton< bs_cfg_p > cfg;

struct bs_config {
	bs_cfg_p::vstr_t operator [] (const char *e);
};

}	//namespace blue_sky

#endif // BS_CONFIG_PARSER_H

