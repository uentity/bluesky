/// @file
/// @author uentity
/// @date 05.08.2016
/// @brief Misc helper functions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "common.h"
#include "detail/str_utils.h"

namespace blue_sky {

//! \brief get time function
BS_API std::string gettime();

BS_API std::string system_message(int err_code);
BS_API std::string last_system_message();

}	//end of blue_sky namespace

