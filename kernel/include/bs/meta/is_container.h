/// @file
/// @author uentity
/// @date 23.10.2017
/// @brief Trait to test if given type is container-like
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "../meta.h"

#include <iterator>

namespace blue_sky::meta {

///////////////////////////////////////////////////////////////////////////////
//  Test if given type is container-like (can be iterated)
//
template<typename T, typename = void>
struct is_container : std::false_type {};

template<typename T>
struct is_container<T, std::void_t<
	decltype(std::begin(std::declval<T>())),
	decltype(std::end(std::declval<T>())),
	decltype(std::size(std::declval<T>()))
>> : public std::true_type {};

template<typename T> using is_container_t = is_container<remove_cvref_t<T>>;
template<typename T> inline constexpr auto is_container_v = is_container_t<T>::value;

///////////////////////////////////////////////////////////////////////////////
//  Test if given type is map-like (container that has `mapped_type`)
//
template<typename T, typename = void>
struct is_map : std::false_type { using mapped_type = void; };

template<typename T>
struct is_map<T, std::enable_if_t<is_container_v<T>, std::void_t<typename T::mapped_type>>> : std::true_type {
	using mapped_type = typename T::mapped_type;
};

template<typename T> using is_map_t = is_map<remove_cvref_t<T>>;
template<typename T> using mapped_type = typename is_map_t<T>::mapped_type;
template<typename T> inline constexpr auto is_map_v = is_map_t<T>::value;

} // blue_sky

