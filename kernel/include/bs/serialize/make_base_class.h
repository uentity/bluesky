/// @file
/// @author uentity
/// @date 01.07.2019
/// @brief A wrapper that produce cereal::base_class<T> (or virtual_base_class<T>) with all considerations
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "object_formatter.h"
#include <cereal/types/base_class.hpp>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(detail)

/// Checks if an archive defines `custom_node_serialization = true`
template<typename A, typename = void>
struct custom_node_serialization : std::false_type {};

template<typename A>
struct custom_node_serialization< A, std::void_t<decltype(A::custom_node_serialization)> > :
	std::integral_constant<bool, A::custom_node_serialization> {};

template<typename A>
inline constexpr auto custom_node_serialization_v =
custom_node_serialization< cereal::traits::detail::decay_archive<A> >::value;

NAMESPACE_END(detail)

template<typename Base, typename Derived, typename Archive>
inline auto make_base_class(Archive& ar, Derived const* derived) {
	if constexpr(std::is_same_v<Base, objbase> || std::is_same_v<Base, objnode>) {
		// get registered formatter for given object
		bool stores_node = true;
		if(auto frm = get_obj_formatter(static_cast<objbase const*>(derived))) {
			// formatter registered => we're in process of tree save/load via Tee FS archive =>
			// objbase metadata is already saved
			if constexpr(std::is_same_v<Base, objbase>)
				return cereal::base_class<Base>(static_cast<Derived const*>(nullptr));
			stores_node = frm->stores_node;
		}
		return cereal::base_class<Base>(stores_node ? derived : nullptr);
	}
	return cereal::base_class<Base>(derived);
}

template<typename Base, typename Derived, typename Archive>
inline auto make_virtual_base_class(Archive& ar, Derived const* derived) {
	if constexpr(std::is_same_v<Base, objbase> || std::is_same_v<Base, objnode>) {
		// get registered formatter for given object
		bool stores_node = true;
		if(auto frm = get_obj_formatter(static_cast<objbase const*>(derived))) {
			// formatter registered => we're in process of tree save/load via Tee FS archive =>
			// objbase metadata is already saved
			if constexpr(std::is_same_v<Base, objbase>)
				return cereal::virtual_base_class<Base>(static_cast<Derived const*>(nullptr));
			stores_node = frm->stores_node;
		}
		return cereal::virtual_base_class<Base>(stores_node ? derived : nullptr);
	}
	return cereal::virtual_base_class<Base>(derived);
}

NAMESPACE_END(blue_sky)
