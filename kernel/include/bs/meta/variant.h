/// @date 02.09.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../meta.h"

#include <caf/detail/type_list.hpp>

#include <variant>

/// adds nessessary specializations to `std` namespace to make `std::visit()` work
/// [CAUTION] this may be not allowed by standard yet
#define BS_ALLOW_VISIT(...)                                                                          \
namespace std {                                                                                      \
template<> struct variant_size<__VA_ARGS__> : variant_size<typename __VA_ARGS__::variant>            \
{};                                                                                                  \
template<size_t Np>                                                                                  \
struct variant_alternative<Np, __VA_ARGS__> : variant_alternative<Np, typename __VA_ARGS__::variant> \
{};                                                                                                  \
template<> struct hash<__VA_ARGS__> : hash<typename __VA_ARGS__::variant>                            \
{};                                                                                                  \
}

namespace blue_sky::meta {

/// trait to detect types derived from std::variant
template<typename V> struct is_variant {
	template< typename... Ts >
	static constexpr auto test(const std::variant<Ts...>&) { return caf::detail::type_list<Ts...>{}; }

	template<typename T>
	static constexpr auto get(int) -> decltype(test(std::declval<T>()), bool{}) { return true; }

	template<typename>
	static constexpr bool get(...) { return false; }

	// true if V is std::variant or derived from it
	static constexpr bool value = get<remove_cvref_t<V>>(0);

	static constexpr auto get_ts() {
		if constexpr(value)
			return decltype(test(std::declval<V>())){};
		else
			return caf::detail::type_list<>{};
	}

	// obtain alternative types that V can carry
	using types = decltype(get_ts());
};

template<typename V> constexpr bool is_variant_v = is_variant<V>::value;

/// check if variant V can have given alternative type T
template<typename V, typename T>
constexpr bool can_hold_alternative() {
	using trait = is_variant<V>;
	static_assert(trait::value, "V is not a variant");
	return caf::detail::tl_contains<typename trait::types, T>::value;
}

/// get index of first element of type T in V
template<typename V, typename T>
constexpr int alternative_index() {
	using trait = is_variant<V>;
	static_assert(trait::value, "V is not a variant");
	return caf::detail::tl_index_of<typename trait::types, T>::value;
}

} // eof blue_sky::meta
