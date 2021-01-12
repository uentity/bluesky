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

/// API macro definitions
#include "setup_common_api.h"
#include BS_SETUP_PLUGIN_API()

/// common includes
#include <type_traits>
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <string_view>

#include "fwd.h"
#include "meta.h"
#include "type_info.h"
#include "detail/scope_guard.h"

NAMESPACE_BEGIN(blue_sky)

/// for kernel subsystems singletons
template< class T >
class singleton {
public:
	static T& Instance();
};

using meta::identity;

NAMESPACE_END(blue_sky)
