/// @file
/// @author uentity
/// @date 27.02.2018
/// @brief BS tree errors enum
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "../error.h"

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

enum class TreeError {
	OK = 0,
	KeyMismatch
};
BS_API std::error_code make_error_code(TreeError);

NAMESPACE_END(blue_sky) NAMESPACE_END(tree)

BS_REGISTER_ERROR_ENUM(blue_sky::tree::TreeError)

