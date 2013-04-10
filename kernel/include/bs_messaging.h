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

#ifndef _BS_MESSAGING_H
#define _BS_MESSAGING_H

#include "bs_common.h"
#include "bs_command.h"
#include "bs_imessaging.h"

#include <map>

#define BS_SIGNAL_RANGE(T) \
(blue_sky::bs_messaging::sig_range_t(T::__bssg_begin__ + 1, T::__bssg_end__))

namespace blue_sky {

class bs_messaging;
typedef smart_ptr< bs_imessaging, true > sp_mobj;

/*-----------------------------------------------------------------
 * Slot object represents an action to be taken on particular signal
 *----------------------------------------------------------------*/
class BS_API bs_slot : public bs_refcounter {
public:
	typedef smart_ptr< bs_slot, true > sp_slot;

	virtual void execute(const sp_mobj& sender, int signal_code, const sp_obj& param) const = 0;

protected:
	void dispose() const;
};

//! type of combase::sp_com
typedef bs_slot::sp_slot sp_slot;

/*-----------------------------------------------------------------
 * Signal is a mechanism for delayed slots calling
 *----------------------------------------------------------------*/
class BS_API bs_signal : public bs_refcounter //: public combase
{
	friend class objbase;

public:
	//! type of blue-sky smart pointer of command
	typedef smart_ptr< bs_signal, true > sp_signal;

	//signal constructor
	bs_signal(int signal_code);
	//dalyed initialization
	void init(int signal_code) const;
	//check if this signal is binded to a valid sender
	//bool sender_binded() const;
	//check signal code
	int get_code() const;

	// if sender != NULL then slot will be activated only for given sender
	bool connect(const sp_slot& slot, const sp_mobj& sender = NULL) const;
	bool disconnect(const sp_slot& slot) const;
	ulong num_slots() const;

	//call slots, connected to this signal
	void fire(const sp_mobj& sender = NULL, const sp_obj& param = NULL) const;

private:
	class signal_impl;
	smart_ptr< signal_impl, false > pimpl_;
	typedef lsmart_ptr< smart_ptr< signal_impl, false > > lpimpl;

//	sp_com execute();
//	void unexecute();

	//destruction method
	void dispose() const;
};

typedef bs_signal::sp_signal sp_signal;

/*-----------------------------------------------------------------
 * bs_messaging is actually a collection of signals
 *----------------------------------------------------------------*/
class BS_API bs_messaging : public bs_imessaging {
public:

	//signals map type - used also inside kernel
	typedef std::map< int, sp_signal > bs_signals_map;

	//signals list starter
	enum singal_codes {
		__bssg_end__ = 1
	};

	typedef std::pair< int, int > sig_range_t;

	virtual bool subscribe(int signal_code, const sp_slot& slot) const;
	virtual bool unsubscribe(int signal_code, const sp_slot& slot) const;
	virtual ulong num_slots(int signal_code) const;
	virtual bool fire_signal(int signal_code, const sp_obj& param = sp_obj (NULL)) const;
	std::vector< int > get_signal_list() const;

	//bs_messaging& operator=(const bs_messaging& lhs);

	//default ctor - doesn't add any signals
	bs_messaging();
	//ctor that adds all signals within given half-open range
	bs_messaging(const sig_range_t& sig_range);
	//copy ctor - copies signals from source object
	bs_messaging(const bs_messaging&);

	//signals list manipulation
	virtual bool add_signal(int signal_code);
	virtual bool remove_signal(int signal_code);
	ulong add_signal(const sig_range_t& sr);

protected:
	//swaps 2 bs_messaging objects
	void swap(bs_messaging& rhs);

	//default dispose method
	virtual void dispose() const;

private:
	//signals map
	bs_signals_map signals_;
};

}	//end of namespace blue_sky

#endif
