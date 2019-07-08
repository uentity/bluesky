/// @file
/// @author uentity
/// @date 01.07.2019
/// @brief A wrapper that produce cereal::base_class<T> (or virtual_base_class<T>) with all considerations
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../common.h"
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
	if constexpr(std::is_same_v<Base, tree::node> && detail::custom_node_serialization_v<Archive>) {
		// ask if an Archive has custom node serialization for this object
		return cereal::base_class<Base>(ar.will_serialize_node(derived) ? nullptr : derived);
	}
	return cereal::base_class<Base>(derived);
}

template<typename Base, typename Derived, typename Archive>
inline auto make_virtual_base_class(Archive& ar, Derived const* derived) {
	if constexpr(std::is_same_v<Base, tree::node> && detail::custom_node_serialization_v<Archive>) {
		// ask if an Archive has custom node serialization for this object
		return cereal::virtual_base_class<Base>(ar.will_serialize_node(derived) ? nullptr : derived);
	}
	return cereal::virtual_base_class<Base>(derived);
}

NAMESPACE_END(blue_sky)
