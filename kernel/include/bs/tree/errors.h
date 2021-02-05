/// @file
/// @author uentity
/// @date 19.09.2018
/// @brief Define error codes for BS tree
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/error.h>

NAMESPACE_BEGIN(blue_sky::tree)

enum class Error {
	OK = 0,

	// link errors
	EmptyData,
	EmptyInode,
	NotANode,
	LinkExpired,
	UnboundSymLink,
	LinkBusy,
	NoFusionBridge,
	WrongLinkCast,

	// node errors
	KeyMismatch,
	WrongOrderSize,

	// Tree FS errors
	EmptyPath,
	PathNotExists,
	PathNotDirectory,
	CantReadFile,
	CantWriteFile,
	LinkWasntStarted,
	NodeWasntStarted,
	MissingFormatter,
	CantMakeFilename
};

BS_API std::error_code make_error_code(Error);

NAMESPACE_END(blue_sky::tree)

BS_REGISTER_ERROR_ENUM(blue_sky::tree::Error)

