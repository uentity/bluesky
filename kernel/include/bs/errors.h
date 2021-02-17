/// @author Alexander Gagarin (@uentity)
/// @date 17.02.2021
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include <system_error>

#define BS_REGISTER_ERROR_ENUM(E) \
namespace std { template<> struct is_error_code_enum< E > : true_type {}; }

NAMESPACE_BEGIN(blue_sky)

enum class Error {
	// generic OK & not OK error codes
	OK = 0,
	Happened = 1,
	Undefined = 2, // will transform to OK or Happened depending on quiet status

	TrEmptyTarget
};

BS_API std::error_code make_error_code(Error);

NAMESPACE_END(blue_sky)

BS_REGISTER_ERROR_ENUM(blue_sky::Error)
