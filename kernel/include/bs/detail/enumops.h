/// @file
/// @author uentity
/// @date 13.11.2017
/// @brief Operator overloading for enums that makes it possible to convert to integral types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <type_traits>
#include <utility>

namespace blue_sky {
namespace detail {

template<typename T, typename R = void>
using enable_if_enum_t = std::enable_if_t<std::is_enum_v<std::decay_t<T>>, R>;

} // eof namespace blue_sky::detail

///////////////////////////////////////////////////////////////////////////////
//  extract enum underlying type value
//
// allows to optionally specify return type
template<typename R = void, typename T, typename = detail::enable_if_enum_t<T>>
constexpr auto enumval(T a) {
	using R_ = std::conditional_t<std::is_same_v<R, void>, std::underlying_type_t<T>, R>;
	return static_cast<R_>(a);
}

namespace allow_enumops {
///////////////////////////////////////////////////////////////////////////////
//  unary ops
//
template<typename T>
constexpr auto operator~(T a) -> detail::enable_if_enum_t<T, T> {
	return T(~enumval(a));
}

///////////////////////////////////////////////////////////////////////////////
//  binary enum - enum ops
//
template<typename T>
constexpr auto operator|(T a, T b) -> detail::enable_if_enum_t<T, T> {
	return static_cast<T>(enumval(a) | enumval(b));
}

template<typename T>
constexpr auto operator&(T a, T b) -> detail::enable_if_enum_t<T, T> {
	return static_cast<T>(enumval(a) & enumval(b));
}

template<typename T>
constexpr auto operator^(T a, T b) -> detail::enable_if_enum_t<T, T> {
	return static_cast<T>(enumval(a) ^ enumval(b));
}

///////////////////////////////////////////////////////////////////////////////
//  binary enum - underlying type ops
//
template<typename T>
constexpr auto operator|(T a, std::underlying_type_t<T> b) -> detail::enable_if_enum_t<T, T> {
	return static_cast<T>(enumval(a) | b);
}

template<typename T>
constexpr auto operator&(T a, std::underlying_type_t<T> b) -> detail::enable_if_enum_t<T, T> {
	return static_cast<T>(enumval(a) & b);
}

template<typename T>
constexpr auto operator^(T a, std::underlying_type_t<T> b) -> detail::enable_if_enum_t<T, T> {
	return static_cast<T>(enumval(a) ^ b);
}

///////////////////////////////////////////////////////////////////////////////
//  in-place lhs modification
//
template<typename T, typename U>
constexpr auto operator|=(T& a, U b) -> detail::enable_if_enum_t<T, T&> {
	return a = a | b;
}

template<typename T, typename U>
constexpr auto operator&=(T& a, U b) -> detail::enable_if_enum_t<T, T&> {
	return a = a & b;
}

template<typename T, typename U>
constexpr auto operator^=(T& a, U b) -> detail::enable_if_enum_t<T, T&> {
	return a = a ^ b;
}

}} /* namespace blue_sky */

