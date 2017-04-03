/// @file
/// @author uentity
/// @date 26.06.2009
/// @brief Main macro definitions to ease defining new BlueSky object
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

//#include <boost/preprocessor/tuple/rem.hpp>

//! BlueSky kernel instance getter macro
#define BS_KERNEL blue_sky::give_kernel::Instance()

// shortcut for quick declaration of shared ptr to BS object
#define BS_SP(T) std::shared_ptr< T >

