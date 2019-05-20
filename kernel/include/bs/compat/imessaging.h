/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Abstract base class for classes that supports signals/slots
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once
#include "../setup_common_api.h"
#include "../fwd.h"

#include <vector>
#include <memory>
#include <any>

//--------------- Signals macro-----------------------------------------------------------------------
namespace blue_sky {

class BS_API bs_imessaging {
public:

	virtual bool subscribe(int signal_code, const std::shared_ptr< bs_slot >& slot) const = 0;
	virtual bool unsubscribe(int signal_code, const std::shared_ptr< bs_slot >& slot) const = 0;
	virtual ulong num_slots(int signal_code) const = 0;
	virtual bool fire_signal(int signal_code, std::any param = {}, const sp_cobj& sender = nullptr) const = 0;
	virtual std::vector< int > get_signal_list() const = 0;
	// virtual dtor
	virtual ~bs_imessaging();
};

typedef std::shared_ptr< const bs_imessaging > sp_mobj;

}	//end of blue_sky namespace

