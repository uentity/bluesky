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
template<> struct ::blue_sky::allow_enumops< __VA_ARGS__ > : std::true_type {};

namespace blue_sky {

template<typename T>
struct allow_enumops : std::false_type {};

///////////////////////////////////////////////////////////////////////////////
//  unary ops
//
template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T operator~(T a) {
	return T(~std::underlying_type_t<T>(a));
}

///////////////////////////////////////////////////////////////////////////////
//  binary enum - enum ops
//
template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T operator|(T a, T b) {
	return T(std::underlying_type_t<T>(a) | std::underlying_type_t<T>(b));
}

template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T operator&(T a, T b) {
	return T(std::underlying_type_t<T>(a) & std::underlying_type_t<T>(b));
}

template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T operator^(T a, T b) {
	return T(std::underlying_type_t<T>(a) ^ std::underlying_type_t<T>(b));
}

template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T& operator|=(T& a, T b) {
	return (T&)((std::underlying_type_t<T>&)a |= std::underlying_type_t<T>(b));
}

template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T& operator&=(T& a, T b) {
	return (T&)((std::underlying_type_t<T>&)a &= std::underlying_type_t<T>(b));
}

template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T& operator^=(T& a, T b) {
	return (T&)((std::underlying_type_t<T>&)a ^= std::underlying_type_t<T>(b));
}

///////////////////////////////////////////////////////////////////////////////
//  binary enum - underlying type ops
//
template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T operator|(T a, std::underlying_type_t<T> b) {
	return T(std::underlying_type_t<T>(a) | b);
}

template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T operator&(T a, std::underlying_type_t<T> b) {
	return T(std::underlying_type_t<T>(a) & b);
}

template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T operator^(T a, std::underlying_type_t<T> b) {
	return T(std::underlying_type_t<T>(a) ^ b);
}

template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T& operator|=(T& a, std::underlying_type_t<T> b) {
	return (T&)((std::underlying_type_t<T>&)a |= b);
}

template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T& operator&=(T& a, std::underlying_type_t<T> b) {
	return (T&)((std::underlying_type_t<T>&)a &= b);
}

template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr T& operator^=(T& a, std::underlying_type_t<T> b) {
	return (T&)((std::underlying_type_t<T>&)a ^= b);
}

///////////////////////////////////////////////////////////////////////////////
//  extract enum underlying type value
//
template<class T, typename = std::enable_if_t<std::is_enum<T>::value && allow_enumops<T>::value>>
constexpr auto enumval(T a) {
	return static_cast<std::underlying_type_t<T>>(a);
}

} /* namespace blue_sky */

