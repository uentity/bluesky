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

#define BS_ALLOW_ENUMOPS(...) \
template<> struct blue_sky::enumops_enabled< blue_sky::__VA_ARGS__ > : std::true_type {};

namespace blue_sky {

/// provide specialization of this to enable ops	
template<typename T> struct enumops_enabled : std::false_type {};

namespace detail {

template<typename T>
inline constexpr auto enumops_enabled_v = enumops_enabled<T>::value;

template<typename T>
static constexpr bool enumops_allowed(...) { return false; }

template<typename T>
static constexpr auto enumops_allowed(int) -> std::enable_if_t<enumops_enabled_v<T>, bool> {
	return std::is_enum_v<T>;
}

template<typename T, typename = void> struct underlying_type { using type = T; };
template<typename T> using underlying_type_t = typename underlying_type<T>::type;

template<typename T>
struct underlying_type<T, std::enable_if_t<std::is_enum_v<T>>> {
	using type = std::underlying_type_t<T>;
};

} // eof namespace blue_sky::detail

template<bool Force, typename T, typename R = void>
using if_enumops_allowed = std::enable_if_t<Force | detail::enumops_allowed<T>(0), R>;

///////////////////////////////////////////////////////////////////////////////
//  extract enum underlying type value
//
// allows to optionally specify return type
template<
	typename R = void, typename T,
	typename = std::enable_if_t<std::is_enum_v<T> || std::is_integral_v<T>>
>
constexpr auto enumval(T a) {
	using R_ = typename std::conditional_t<std::is_same_v<R, void>, detail::underlying_type_t<T>, R>;
	return static_cast<R_>(a);
}

///////////////////////////////////////////////////////////////////////////////
//  unary ops
//
template<bool Force = false, typename T>
constexpr auto operator~(T a) -> if_enumops_allowed<Force, T, T> {
	return T(~enumval(a));
}

///////////////////////////////////////////////////////////////////////////////
//  binary enum - enum ops
//
template<bool Force = false, typename T>
constexpr auto operator|(T a, T b) -> if_enumops_allowed<Force, T, T> {
	return static_cast<T>(enumval(a) | enumval(b));
}

template<bool Force = false, typename T>
constexpr auto operator&(T a, T b) -> if_enumops_allowed<Force, T, T> {
	return static_cast<T>(enumval(a) & enumval(b));
}

template<bool Force = false, typename T>
constexpr auto operator^(T a, T b) -> if_enumops_allowed<Force, T, T> {
	return static_cast<T>(enumval(a) ^ enumval(b));
}

///////////////////////////////////////////////////////////////////////////////
//  binary enum - underlying type ops
//
template<bool Force = false, typename T>
constexpr auto operator|(T a, std::underlying_type_t<T> b) -> if_enumops_allowed<Force, T, T> {
	return static_cast<T>(enumval(a) | b);
}

template<bool Force = false, typename T>
constexpr auto operator&(T a, std::underlying_type_t<T> b) -> if_enumops_allowed<Force, T, T> {
	return static_cast<T>(enumval(a) & b);
}

template<bool Force = false, typename T>
constexpr auto operator^(T a, std::underlying_type_t<T> b) -> if_enumops_allowed<Force, T, T> {
	return static_cast<T>(enumval(a) ^ b);
}

///////////////////////////////////////////////////////////////////////////////
//  in-place lhs modification
//
template<typename T, typename U>
constexpr auto operator|=(T& a, U b) -> if_enumops_allowed<false, T, T&> {
	return a = a | b;
}

template<typename T, typename U>
constexpr auto operator&=(T& a, U b) -> if_enumops_allowed<false, T, T&> {
	return a = a & b;
}

template<typename T, typename U>
constexpr auto operator^=(T& a, U b) -> if_enumops_allowed<false, T, T&> {
	return a = a ^ b;
}

/*-----------------------------------------------------------------------------
 *  add `using namespace allow_enumops` to enable ops for all enums in a scope
 *-----------------------------------------------------------------------------*/
namespace allow_enumops {

template<typename T, typename R = void>
using if_enumops_allowed = std::enable_if_t<!detail::enumops_allowed<T>(0), R>;

///////////////////////////////////////////////////////////////////////////////
//  unary ops
//
template<bool Force = false, typename T>
constexpr auto operator~(T a) -> if_enumops_allowed<T, T> {
	return blue_sky::operator~<true>(a);
}

///////////////////////////////////////////////////////////////////////////////
//  binary enum - enum ops
//
template<bool Force = false, typename T>
constexpr auto operator|(T a, T b) -> if_enumops_allowed<T, T> {
	return blue_sky::operator|<true>(a, b);
}

template<bool Force = false, typename T>
constexpr auto operator&(T a, T b) -> if_enumops_allowed<T, T> {
	return blue_sky::operator&<true>(a, b);
}

template<bool Force = false, typename T>
constexpr auto operator^(T a, T b) -> if_enumops_allowed<T, T> {
	return blue_sky::operator^<true>(a, b);
}

///////////////////////////////////////////////////////////////////////////////
//  binary enum - underlying type ops
//
template<bool Force = false, typename T>
constexpr auto operator|(T a, std::underlying_type_t<T> b) -> if_enumops_allowed<T, T> {
	return blue_sky::operator|<true>(a, b);
}

template<bool Force = false, typename T>
constexpr auto operator&(T a, std::underlying_type_t<T> b) -> if_enumops_allowed<T, T> {
	return blue_sky::operator&<true>(a, b);
}

template<bool Force = false, typename T>
constexpr auto operator^(T a, std::underlying_type_t<T> b) -> if_enumops_allowed<T, T> {
	return blue_sky::operator^<true>(a, b);
}

///////////////////////////////////////////////////////////////////////////////
//  in-place lhs modification
//
template<typename T, typename U>
constexpr auto operator|=(T& a, U b) -> if_enumops_allowed<T, T&> {
	return a = a | b;
}

template<typename T, typename U>
constexpr auto operator&=(T& a, U b) -> if_enumops_allowed<T, T&> {
	return a = a & b;
}

template<typename T, typename U>
constexpr auto operator^=(T& a, U b) -> if_enumops_allowed<T, T&> {
	return a = a ^ b;
}

} // eof blue_sky::allow_enumops

} /* namespace blue_sky */

