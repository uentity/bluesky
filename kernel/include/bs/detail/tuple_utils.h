/// @file
/// @author uentity
/// @date 26.08.2016
/// @brief Misc std::tuple related helpers
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <tuple>
#include <utility>
#include <type_traits>

namespace blue_sky {

/*-----------------------------------------------------------------
 * Constructs tuple with references to given arguments
 * lvalue references in arguments are passed as is
 * rvalue objects in arguments are passed as rvalue references
 *----------------------------------------------------------------*/
template <typename T>
constexpr decltype(auto) forward_tuple(T&& t) {
    return apply(std::forward_as_tuple, std::forward<T>(t));
}

/*-----------------------------------------------------------------
 * Helper to generaate integer sequence in given bounds
 *----------------------------------------------------------------*/
template< std::size_t From, std::size_t To >
struct make_bounded_sequence {
	template< std::size_t offset, std::size_t... Ints >
	decltype(auto) static constexpr offset_sequence(
		std::integral_constant< std::size_t, offset >, std::index_sequence< Ints... >
	) {
		return std::index_sequence< Ints + offset... >{};
	}

	static constexpr auto value = offset_sequence(
		std::integral_constant< std::size_t, From >(),
		std::make_index_sequence< To - From >{}
	);
};

/*-----------------------------------------------------------------
 * Helper to extract range of tuple args as new tuple
 *----------------------------------------------------------------*/
template< typename Tup, std::size_t... index >
constexpr decltype(auto) subtuple_helper(Tup&& tup, std::index_sequence<index...>) {
	// use explicit tuple construction in order to save exact information about elem types
	return std::tuple< std::tuple_element_t< index, std::decay_t< Tup > >... >(
		std::get< index >(std::forward< Tup >(tup))...
	);
}

template<
	typename Tup, std::size_t From = 0,
	std::size_t To = std::tuple_size< std::decay_t< Tup > >::value
>
constexpr decltype(auto) subtuple(
	Tup&& tup,
	std::integral_constant< std::size_t, From > from = std::integral_constant< std::size_t, 0 >(),
	std::integral_constant< std::size_t, To > to = std::integral_constant<
		std::size_t, std::tuple_size< std::decay_t< Tup > >::value
	>()
) {
	return subtuple_helper(
		std::forward<Tup>(tup),
		make_bounded_sequence< From, To >::value
	);
}

} /* namespace blue_sky */

