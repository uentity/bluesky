/// @file
/// @author uentity
/// @date 10.06.2019
/// @brief `propdict` & `propbook` serialization tweaks
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../propdict.h"
#include "base_types.h"

#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/types/map.hpp>

NAMESPACE_BEGIN(cereal)

/// difference from Cereal-bundled map load is that we DO NOT clear target map
template<typename Archive, typename Key, typename MappedType>
auto load(Archive& ar, std::map<Key, MappedType, std::less<>>& map) -> std::enable_if_t<
	std::is_same_v<MappedType, blue_sky::prop::property> ||
	std::is_same_v<MappedType, blue_sky::prop::propdict>
> {
	using Propmap = std::map<Key, MappedType, std::less<>>;

	cereal::size_type size;
	ar( cereal::make_size_tag( size ) );

	[[maybe_unused]] auto hint = map.begin();
	for(size_t i = 0; i < size; ++i) {
		typename Propmap::key_type key;
		MappedType value;

		ar( cereal::make_map_item(key, value) );
		if constexpr(std::is_same_v<MappedType, blue_sky::prop::property>)
			hint = map.insert_or_assign(hint, std::move(key), std::move(value));
		else {
			auto I = map.try_emplace(std::move(key), std::move(value));
			if(!I.second) I.first->second.merge_props(std::move(value));
		}
	}
}

NAMESPACE_END(cereal)
