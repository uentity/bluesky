/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "bs_type_info.h"

namespace blue_sky {
	class nil {};
	
	bs_type_info::bs_type_info()
	{
		pinfo_ = &typeid(nil);
		assert(pinfo_);
	}

	bool bs_type_info::is_nil() const {
		return (pinfo_ == &typeid(nil));
	}
}
