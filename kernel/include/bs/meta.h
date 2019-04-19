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

} // eof blue_sky::meta
