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

#ifndef BS_CUBE_H_
#define BS_CUBE_H_

#include "bs_object_base.h"
#include "bs_command.h"
//#include "bs_tree.h"

#include <sstream>

#ifdef BSPY_EXPORTING_PLUGIN
#include "py_bs_tree.h"
#include "py_bs_object_base.h"
#include "py_bs_command.h"
#include "bs_plugin_common.h"
#endif

namespace blue_sky {
	 namespace python {
			void register_obj();
	 }

	 class BS_API_PLUGIN bs_cube : public objbase
	{
		friend void python::register_obj();
		static int py_factory_index;

	public:
		~bs_cube();
		void test();
		int get_py_factory_index();

		bs_cube &operator=(const bs_cube &src) {
			var_ = src.var_;
			logname.str("");
			return *this;
		}

	private:
		int var_;
		std::ostringstream logname;

		BLUE_SKY_TYPE_DECL(bs_cube);
	};

	class BS_API_PLUGIN cube_command : public objbase, public combase
	{
	public:

		sp_com execute();
		void unexecute();
		void test();

		void dispose() const {
			delete this;
		}

		BLUE_SKY_TYPE_DECL(cube_command)
	};
}

#endif
