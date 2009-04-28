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

protected:

	//signals list manipulation
	virtual bool add_signal(int signal_code) = 0;
	virtual bool remove_signal(int signal_code) = 0;
};

}	//end of blue_sky namespace

#endif
