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
	// special value that is not an error indicator (also means 'OK')
	// intended to be used for ex. by `fusion_iface` to indicate that object is fully loaded
	// in single call to `populate()` or `pull_data()`
	OKOK,

	EmptyData,
	EmptyInode,
	NotANode,
	LinkExpired,
	UnboundSymLink,
	LinkBusy,
	NoFusionBridge,
	KeyMismatch,

	// Tree FS related errors
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

