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

using meta::identity;

/// for kernel subsystems singletons
template< class T >
class singleton {
public:
	static T& Instance();
};

/// can be passed as callback that does nothing
inline constexpr auto noop = [](auto&&...) {};
using noop_t = decltype(noop);

template<typename R>
constexpr auto noop_r() {
	return [](auto&&...) -> R { return {}; };
}

template<typename R>
constexpr auto noop_r(R res) {
	return [res = std::move(res)](auto&&...) -> R { return res; };
}

template<typename R> using noop_r_t = decltype( noop_r(std::declval<R>()) );

template<bool Res = true>
inline constexpr auto bool_noop = [](auto&&...) { return Res; };
inline constexpr auto noop_true = bool_noop<true>;
inline constexpr auto noop_false = bool_noop<false>;

NAMESPACE_END(blue_sky)
