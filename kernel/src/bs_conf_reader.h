// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

/*!
 * \file bs_conf_reader.h
 * \brief Contains blue-sky library configs reader.
 * \author Nikonov Maxim aka no_NaMe
*/

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

		const conf_elem &get(int i) const;
		conf_elem &get(int i);

	private:
		conf_elem_array carray;

		BLUE_SKY_TYPE_DECL(bs_conf_reader);
	};

	typedef bs_conf_reader::sp_conf_reader sp_conf_reader;

}

#endif // _BS_CONF_READER
