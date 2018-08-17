/// @file
/// @author uentity
/// @date 28.04.2016
/// @brief Common includes and definitions for BlueSky
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#if !defined(NAMESPACE_BEGIN)
#  define NAMESPACE_BEGIN(name) namespace name {
#endif
#if !defined(NAMESPACE_END)
#  define NAMESPACE_END(name) }
#endif

// prevent warnings about macro redeifinition - include Python.h
// at the very beginning
#if defined(BSPY_EXPORTING) || defined(BSPY_EXPORTING_PLUGIN)
	#if defined(_MSC_VER) && _MSC_VER >= 1800
		#define HAVE_ROUND 1
	#endif
	#ifdef _DEBUG
		// disable linking to pythonX_d.lib on Windows in debug mode
		#undef _DEBUG
		#include <Python.h>
		#define _DEBUG
	#else
		#include <Python.h>
	#endif
#endif

// detect C++17 support
#if !defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#  if __cplusplus >= 201402L
#    define BS_CPP14
#    if __cplusplus > 201402L /* Temporary: should be updated to >= the final C++17 value once known */
#      define BS_CPP17
#    endif
#  endif
#elif defined(_MSC_VER)
// MSVC sets _MSVC_LANG rather than __cplusplus (supposedly until the standard is fully implemented)
#  if _MSVC_LANG >= 201402L
#    define BS_CPP14
#    if _MSVC_LANG > 201402L && _MSC_VER >= 1910
#      define BS_CPP17
#    endif
#  endif
#endif

// local BS includes
// API macro definitions
#include "setup_common_api.h"
#include BS_SETUP_PLUGIN_API()

// common includes - used almost everywhere
#include <type_traits>
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <iosfwd>
#include <algorithm>
#ifdef BS_CPP17
	#include <optional>
	#define bs_optional std::optional
#else
	#include <boost/optional.hpp>
	#define bs_optional boost::optional
#endif

#include "fwd.h"
#include "type_info.h"
#include "assert.h"
#include "detail/args.h"
#include "detail/scope_guard.h"

NAMESPACE_BEGIN(blue_sky)

/// BlueSky singleton template
template< class T >
class singleton {
public:
	static T& Instance();
};

// identity utility template to pass params to templated constructors
template< class T > struct identity { using type = T; };

NAMESPACE_END(blue_sky)

