/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

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
