/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Abstract base class for classes that supports signals/slots
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _BS_IMESSAGING_H
#define _BS_IMESSAGING_H

#include "setup_common_api.h"
#include "bs_fwd.h"
#include "bs_refcounter.h"
#include <vector>

//--------------- Signals macro-----------------------------------------------------------------------
#define BLUE_SKY_SIGNALS_DECL_BEGIN(base) \
public: enum signal_codes { \
	__bssg_begin__ = base::__bssg_end__ - 1,

#define BLUE_SKY_SIGNALS_DECL_END \
	__bssg_end__ };

namespace blue_sky {

class BS_API bs_imessaging : virtual public bs_refcounter {
public:
	//signals list starter
	enum singal_codes {
		__bssg_end__ = 1
	};

	virtual bool subscribe(int signal_code, const smart_ptr< bs_slot, true >& slot) const = 0;
	virtual bool unsubscribe(int signal_code, const smart_ptr< bs_slot, true >& slot) const = 0;
	virtual ulong num_slots(int signal_code) const = 0;
	virtual bool fire_signal(int signal_code, const sp_obj& param) const = 0;
	virtual std::vector< int > get_signal_list() const = 0;
};

}	//end of blue_sky namespace

#endif

