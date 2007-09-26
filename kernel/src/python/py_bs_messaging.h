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

#ifndef _PY_BS_MESSAGING_H
#define _PY_BS_MESSAGING_H

#include <boost/python/wrapper.hpp>

#include "bs_messaging.h"
#include "py_bs_command.h"
#include "bs_common.h"

namespace blue_sky {
namespace python {

class py_objbase;
class py_bs_messaging;

class BS_API python_slot : public bs_slot, public boost::python::wrapper<bs_slot> {
public:
  python_slot() 
		//: spslot(this)		// author: Sergey Miryanov
		{
			spslot = this;
		}

	void execute(const sp_mobj& sender = sp_mobj (NULL), int signal_code = 0, const sp_obj& param = sp_obj (NULL)) const;

	sp_slot spslot;
};

class BS_API py_bs_messaging {
	friend class py_bs_slot;
public:
	py_bs_messaging(const sp_mobj&);
	py_bs_messaging(const py_bs_messaging&);

	bool subscribe(int signal_code, const python_slot& slot) const;
	bool unsubscribe(int signal_code, const python_slot& slot) const;
	ulong num_slots(int signal_code) const;
	bool fire_signal(int signal_code, const py_objbase* param) const;
	std::vector< int > get_signal_list() const;

protected:
	sp_mobj spmsg;

	py_bs_messaging& operator=(const sp_mobj& lhs);
};

}	//namespace blue_sky::python
}	//namespace blue_sky

#endif // _PY_BS_MESSAGING_H
