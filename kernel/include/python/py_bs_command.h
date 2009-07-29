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

#ifndef _PY_BS_COMMAND_H
#define _PY_BS_COMMAND_H

#include "bs_common.h"
#include "bs_command.h"

namespace blue_sky {
namespace python {

class BS_API py_combase {
	 friend class py_kernel;
	 friend class py_bs_slot;

public:
	 //typedef combase::sp_com sp_com;

	 virtual ~py_combase() {}

	 const sp_com &get_spcom();

	 py_combase execute();
	 void unexecute();
	 bool can_unexecute() const;

protected:
	 //py_combase(combase*);
	 py_combase(const sp_com&);

	 sp_com spcom;
};

}	//namespace blue_sky::python
}	//namespace blue_sky

#endif // _PY_BS_COMMAND_H
