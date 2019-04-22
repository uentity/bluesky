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
#include <string_view>

#include "fwd.h"
#include "type_info.h"
#include "detail/scope_guard.h"
#include <object_ptr.hpp>

NAMESPACE_BEGIN(blue_sky)

/// BlueSky singleton template
template< class T >
class singleton {
public:
	static T& Instance();
};

// identity utility template to pass params to templated constructors
template< class T > struct identity { using type = T; };

// object_ptr to be used as functions param
template<typename T> using object_ptr = jss::object_ptr<T>;

NAMESPACE_END(blue_sky)

