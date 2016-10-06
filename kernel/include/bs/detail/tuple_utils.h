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
#include "invoke.h"

namespace blue_sky {

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
 * invoke given function with arguments unpacked from tuple
 *----------------------------------------------------------------*/
// update: this should be available in C++17, so make implementation to match upcoming standard
template< typename F, typename Tuple, size_t... I >
constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
	// missing std::invoke here
	return invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))...);
}

template< typename F, typename Tuple >
constexpr decltype(auto) apply(F&& f, Tuple&& t) {
	return apply_impl(
		std::forward<F>(f), std::forward<Tuple>(t),
		std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{}
	);
}

template<
	typename F, typename Tuple, std::size_t From = 0,
	std::size_t To = std::tuple_size< std::decay_t< Tuple > >::value
>
constexpr decltype(auto) apply_range(
	F&& f, Tuple&& t,
	std::integral_constant< std::size_t, From > = std::integral_constant< std::size_t, 0 >(),
	std::integral_constant< std::size_t, To > = std::integral_constant<
		std::size_t, std::tuple_size< std::decay_t< Tuple > >::value
	>()
) {
	return apply_impl(
		std::forward<F>(f), std::forward<Tuple>(t),
		make_bounded_sequence< From, To >::value
	);
}

//template<typename Func, typename Tup, std::size_t... index>
//decltype(auto) invoke_helper(Func&& func, Tup&& tup, std::index_sequence<index...>) {
//	return func(std::get<index>(std::forward<Tup>(tup))...);
//}
//
//template<typename Func, typename Tup>
//decltype(auto) invoke(Func&& func, Tup&& tup) {
//	constexpr auto Size = std::tuple_size< std::decay_t<Tup> >::value;
//	return invoke_helper(
//		std::forward<Func>(func),
//		std::forward<Tup>(tup),
//		std::make_index_sequence<Size>{}
//	);
//}

//template<
//	typename Func, typename Tup, std::size_t From = 0,
//	std::size_t To = std::tuple_size< std::decay_t< Tup > >::value
//>
//decltype(auto) invoke_range(
//	Func&& func, Tup&& tup,
//	std::integral_constant< std::size_t, From > = std::integral_constant< std::size_t, 0 >(),
//	std::integral_constant< std::size_t, To > = std::integral_constant<
//		std::size_t, std::tuple_size< std::decay_t< Tup > >::value
//	>()
//) {
//	return invoke_helper(
//		std::forward<Func>(func),
//		std::forward<Tup>(tup),
//		make_bounded_sequence< From, To >::value
//	);
//}

/*-----------------------------------------------------------------
 * Helper to extract range of tuple args as new tuple
 *----------------------------------------------------------------*/
template< typename Tup, std::size_t... index >
decltype(auto) subtuple_helper(Tup&& tup, std::index_sequence<index...>) {
	// use explicit tuple construction in order to save exact information about elem types
	return std::tuple< std::tuple_element_t< index, std::decay_t< Tup > >... >(
		std::get< index >(std::forward< Tup >(tup))...
	);
}

template<
	typename Tup, std::size_t From = 0,
	std::size_t To = std::tuple_size< std::decay_t< Tup > >::value
>
decltype(auto) subtuple(
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

