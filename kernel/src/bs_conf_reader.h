/// @file
/// @author Nikonov Maxim aka no_NaMe
/// @date 12.01.2016
/// @brief Contains BlueSky library configs reader
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _BS_CONF_READER
#define _BS_CONF_READER

#include "bs_common.h"
#include "bs_object_base.h"

#include <map>
#include <vector>
#include <string>

namespace blue_sky {

	class BS_API bs_conf_reader : public objbase {
	public:
		typedef smart_ptr< bs_conf_reader, true > sp_conf_reader;

		struct conf_elem {
			//struct value_t {
				//value_t (const char *tname = 0, const char *tvalue = 0);

				//std::string name;
				//std::string value;
				//};
			//typedef std::list<value_t> value_list_t;
			typedef std::map<std::string,std::string> val_map_t;

			std::string lookup_value(const std::string &what) const;
			void add (const std::string &name, const std::string &val);
			//value_list_t lval;
			val_map_t mval;
		};

		typedef std::vector<conf_elem> conf_elem_array;

		void read_file(const char *filename);
		size_t get_length() const;

		const conf_elem &get(size_t i) const;
		conf_elem &get(size_t i);

	private:
		conf_elem_array carray;

		BLUE_SKY_TYPE_DECL(bs_conf_reader);
	};

	typedef bs_conf_reader::sp_conf_reader sp_conf_reader;

}

#endif // _BS_CONF_READER
