/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief BlueSky messaging-related classes declarations
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "../common.h"
#include "../objbase.h"
#include "../type_macro.h"
#include "imessaging.h"

#include <map>

namespace blue_sky {
/*-----------------------------------------------------------------
 * Slot object represents an action to be taken on particular signal
 *----------------------------------------------------------------*/
class BS_API bs_slot {
public:
	typedef std::shared_ptr< bs_slot > sp_slot;

	virtual void execute(sp_mobj sender, int signal_code, sp_obj param) const = 0;
};
using sp_slot = bs_slot::sp_slot;

/*-----------------------------------------------------------------
 * Signal is a mechanism for delayed slots calling
 *----------------------------------------------------------------*/
class BS_API bs_signal : public objbase {
	friend class objbase;

public:
	//! type of blue-sky smart pointer of command
	typedef std::shared_ptr< bs_signal > sp_signal;

	bs_signal(int signal_code);
	~bs_signal();
	// delayed initialization
	void init(int signal_code) const;
	// check signal code
	int get_code() const;

	// if sender != nullptr then slot will be activated only for given sender
	bool connect(const sp_slot& slot, const sp_mobj& sender = nullptr) const;
	bool disconnect(const sp_slot& slot) const;
	ulong num_slots() const;

	//call slots, connected to this signal
	void fire(const sp_mobj& sender = nullptr, const sp_obj& param = nullptr) const;

private:
	class signal_impl;
	std::unique_ptr< signal_impl > pimpl_;

	BS_TYPE_DECL
};

typedef bs_signal::sp_signal sp_signal;

/*-----------------------------------------------------------------
 * bs_messaging is actually a collection of signals
 *----------------------------------------------------------------*/
class BS_API bs_messaging :
	public bs_imessaging, public objbase {
public:

	// signals map type - used also inside kernel
	typedef std::map< int, sp_signal > bs_signals_map;

	// signals list starter
	enum singal_codes {
		__bssg_end__ = 1
	};

	typedef std::pair< int, int > sig_range_t;

	bool subscribe(int signal_code, const sp_slot& slot) const;
	bool unsubscribe(int signal_code, const sp_slot& slot) const;
	ulong num_slots(int signal_code) const;
	bool fire_signal(int signal_code, const sp_obj& param = nullptr) const;
	std::vector< int > get_signal_list() const;

	// default ctor - doesn't add any signals
	bs_messaging();
	// ctor that adds all signals within given range
	bs_messaging(const sig_range_t& sig_range);
	// copy ctor - copies signals from source object
	bs_messaging(const bs_messaging&);

	// signals list manipulation
	virtual bool add_signal(int signal_code);
	virtual bool remove_signal(int signal_code);
	ulong add_signal(const sig_range_t& sr);
	ulong clear();

protected:
	//swaps 2 bs_messaging objects
	void swap(bs_messaging& rhs);

private:
	//signals map
	bs_signals_map signals_;

	BS_TYPE_DECL
};

}	//end of namespace blue_sky

