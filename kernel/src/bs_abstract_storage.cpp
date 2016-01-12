/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "bs_abstract_storage.h"
#include "bs_command.h"
#include "bs_kernel.h"

namespace blue_sky { 
empty_storage::empty_storage(bs_type_ctor_param /*param*/)
{}

empty_storage::empty_storage(const empty_storage& src) : bs_refcounter (src), bs_abstract_storage ()
{
	*this = src;
}

BLUE_SKY_TYPE_STD_CREATE(empty_storage);
BLUE_SKY_TYPE_STD_COPY(empty_storage);
BLUE_SKY_TYPE_IMPL(empty_storage, objbase, "empty_storage", "", "");

} 	//end of blue-sky namespace

