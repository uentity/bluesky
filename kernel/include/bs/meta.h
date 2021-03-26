/// @file
/// @author uentity
/// @date 10.04.2019
/// @brief Misc metaprogramming tools and helpers
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <type_traits>

namespace blue_sky::meta {
/// useful to postpone `static_assert`
template<typename... Ts>
inline constexpr bool static_false = false;

/// tag to denote empty variadic pack
struct empty_t {};

/// C++20 have this in <type_traits>
template<typename T> struct identity { using type = T; };

/// same as `std::forward`, but forward value AS if it has type `AsT`
/// with all type type props copied from `T`
template<typename T, typename AsT>
constexpr decltype(auto) forward_as(typename std::remove_reference_t<T>& value) {
	// copy type props from T to AsT: U = type_props(T) -> AsT
	// 1. add const & volatile if T has corresponding qualifiers
	using U1 = std::conditional_t<std::is_const_v<std::remove_reference_t<T>>, std::add_const_t<AsT>, AsT>;
	using U2 = std::conditional_t<std::is_volatile_v<std::remove_reference_t<T>>, std::add_volatile_t<U1>, U1>;
	// 2. finally add lvalue ref if T is lvalue ref
	using U = std::conditional_t<std::is_lvalue_reference_v<T>, std::add_lvalue_reference_t<U1>, U2>;
	return static_cast<U&&>(value);
}

/// helper for visitors overloading
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

/// until we have C++20
template<typename T> using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

/// remove const/ref qualifiers in contrast to std counterparts
template<typename T, typename U>
inline constexpr bool is_same_uq = std::is_same_v<remove_cvref_t<T>, remove_cvref_t<U>>;

template<typename T, typename U>
inline constexpr bool is_base_of_uq = std::is_base_of_v<T, remove_cvref_t<U>>;

/// extract pure first type from variadic pack (`empty_t` if pack is empty)
template<typename A1 = empty_t, typename... Args>
struct a1 { using type = remove_cvref_t<A1>; };

template<typename... Args> using a1_t = typename a1<Args...>::type;

/// check that first arg (cvref removed) from variadic pack is T (exact match or inherited from)
template<typename T, typename A1 = std::add_pointer_t<T>, typename... Args>
inline constexpr bool a1_is_t = std::conditional_t<
	std::is_class_v<T>, std::is_base_of<T, remove_cvref_t<A1>>, std::is_same<T, remove_cvref_t<A1>>
>::value;

/// same as above, but check if A1 is convertible to T
template<typename T, typename A1 = std::add_pointer_t<T>, typename... Args>
inline constexpr bool a1_convertible_to_t = std::is_convertible_v<remove_cvref_t<A1>, T>;

/// following can be used to narrow greedy perfect forward ctor by excluding copy & move ctors
template<typename T, typename... Args>
inline constexpr bool enable_pf_ctor_v = !(a1_is_t<T, Args...> && sizeof...(Args) == 1);

template<typename T, typename... Args>
using enable_pf_ctor = std::enable_if_t<enable_pf_ctor_v<T, Args...>>;

/// assume that T's ctor var args pack is forwarded to initialize some other type U (for ex. base)
template<typename T, typename U, typename... Args>
inline constexpr bool enable_pf_ctor_to_v = enable_pf_ctor_v<T, Args...> &&
	std::is_constructible_v<U, Args...>;

template<typename T, typename U, typename... Args>
using enable_pf_ctor_to = std::enable_if_t<enable_pf_ctor_to_v<T, U, Args...>>;

/// trait for detecting exactly enum class && not classic enums
template<typename T>
inline constexpr auto is_enum_class_v = std::is_enum_v<remove_cvref_t<T>> &&
	!std::is_convertible_v<remove_cvref_t<T>, int>;

} // eof blue_sky::meta
