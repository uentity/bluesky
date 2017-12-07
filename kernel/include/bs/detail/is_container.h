/// @file
/// @author uentity
/// @date 23.10.2017
/// @brief Trait to test if given type is container-like
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <type_traits>
#include <iterator>

namespace blue_sky {
namespace detail {

template<typename... Ts>
struct is_container_helper {};

} // detail

/// Trait to test if given type is container-like
template<typename T, typename = void>
struct is_container : std::false_type {};

template<typename T>
struct is_container<
	T,
	std::conditional_t<
		false,
		detail::is_container_helper<
			decltype(std::begin(std::declval<std::decay_t<T>>())),
			decltype(std::end(std::declval<std::decay_t<T>>()))
		>,
		void
	>
> : public std::true_type {};

} // blue_sky

