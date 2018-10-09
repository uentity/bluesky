/// @file
/// @author uentity
/// @date 29.09.2018
/// @brief Kernel error codes
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "error.h"

NAMESPACE_BEGIN(blue_sky)

enum class KernelError {
	OK = 0,

	CantLoadDLL,
	CantUnloadDLL,
	CantRegisterType,
	BadTypeDescriptor,
	TypeIsNil,
	TypeAlreadyRegistered,
	CantCreateLogger
};

BS_API std::error_code make_error_code(KernelError);

NAMESPACE_END(blue_sky)

BS_REGISTER_ERROR_ENUM(blue_sky::KernelError)

