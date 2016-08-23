/// @file
/// @author uentity
/// @date 28.04.2016
/// @brief Common includes and definitions for BlueSky
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#define LOKI_FUNCTOR_IS_NOT_A_SMALLOBJECT

// local BS includes
// API macro definitions
#include "setup_common_api.h"
#include BS_SETUP_PLUGIN_API()

#include "fwd.h"
#include "type_info.h"
#include "plugin_common.h"
#include "detail/args.h"

// common includes - used almost everywhere
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <iosfwd>
#include <algorithm>

// third-party libraries
#include <loki/TypeManip.h>

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

//! \defgroup loading_plugins loading plugins - classes for load plugins in blue-sky

#define BLUE_SKY_INIT_PY_FUN \
BS_C_API_PLUGIN void bs_init_py_subsystem()
typedef void (*bs_init_py_fn)();

/// BlueSky singleton template
template< class T >
class singleton {
public:
	static T& Instance();
};

} // eof blue_sky namespace

