/// @file
/// @author uentity
/// @date 28.04.2016
/// @brief Common includes and definitions for BlueSky
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

// prevent warnings about macro redeifinition - include Python.h
// at the very beginning
#if defined(BSPY_EXPORTING) || defined(BSPY_EXPORTING_PLUGIN)
#ifdef _MSC_VER
#define HAVE_ROUND
#endif
#ifdef _DEBUG
	// stop Python from forcing linking to debug library
	#undef _DEBUG
	#include <Python.h>
	#define _DEBUG
#else
	#include <Python.h>
#endif
#endif

// local BS includes
// API macro definitions
#include "setup_common_api.h"
#include BS_SETUP_PLUGIN_API()

#include "fwd.h"
#include "type_info.h"
#include "assert.h"
#include "detail/args.h"

// common includes - used almost everywhere
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <iosfwd>
#include <algorithm>

//#define TO_STR(s) #s //!< To string convertion macro

#if !defined(NAMESPACE_BEGIN)
#  define NAMESPACE_BEGIN(name) namespace name {
#endif
#if !defined(NAMESPACE_END)
#  define NAMESPACE_END(name) }
#endif

namespace blue_sky {

//DECLARE_ERROR_CODES(
//	((no_error,                 "no_error"))
//	((user_defined,             "user_defined"))
//	((unknown_error,            "unknown_error"))
//	((system_error,             "system_error"))
//	((wrong_path,               "wrong_path"))
//	((no_plugins,               "no_plugins"))
//	((no_library,               "no_library"))
//	((no_type,                  "no_type"))
//	((out_of_range,             "out_of_range"))
//	((not_permited_operation,   "not_permited_operation"))
//	((boost_error,              "boost_error"))
//);

/// BlueSky singleton template
template< class T >
class singleton {
public:
	static T& Instance();
};

// identity utility template to pass params to templated constructors
template< class T > struct identity { using type = T; };

} // eof blue_sky namespace

