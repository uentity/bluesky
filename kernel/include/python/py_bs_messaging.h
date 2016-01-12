/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

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
			spslot.lock()->add_ref();
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
