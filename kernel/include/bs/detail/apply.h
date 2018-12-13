/// @file
/// @author uentity
/// @date 28.10.2016
/// @brief Implementation of std::apply
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "tuple_utils.h"
#include <functional>
#define bs_invoke std::invoke

namespace blue_sky { namespace detail {

template< typename F, typename Tuple, size_t... I >
constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
	return bs_invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))...);
}

} // eof namespace detail

/*-----------------------------------------------------------------
 * Same as above but pass given range of arguments
 *----------------------------------------------------------------*/
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
	return blue_sky::detail::apply_impl(
		std::forward<F>(f), std::forward<Tuple>(t),
		make_bounded_sequence< From, To >::value
	);
}

/*-----------------------------------------------------------------
 * Constructs tuple with references to given arguments
 * lvalue references in arguments are passed as is
 * rvalue objects in arguments are passed as rvalue references
 *----------------------------------------------------------------*/
template <typename T>
constexpr decltype(auto) forward_tuple(T&& t) {
    return apply(std::forward_as_tuple, std::forward<T>(t));
}

} // eof namespace blue_sky

