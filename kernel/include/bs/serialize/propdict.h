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
#include "../detail/is_container.h"

#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/types/map.hpp>

NAMESPACE_BEGIN(cereal)

/// difference from Cereal-bundled map load is that we DO NOT clear target map
template<typename Archive, typename Propmap>
auto load(Archive& ar, Propmap& map) -> std::enable_if_t<
	std::is_same_v<blue_sky::meta::mapped_type<Propmap>, blue_sky::prop::property> ||
	std::is_same_v<blue_sky::meta::mapped_type<Propmap>, blue_sky::prop::propdict>
> {
	cereal::size_type size;
	ar( cereal::make_size_tag( size ) );

	auto hint = map.begin();
	for(size_t i = 0; i < size; ++i) {
		typename Propmap::key_type key;
		typename Propmap::mapped_type value;

		ar( cereal::make_map_item(key, value) );
		hint = map.insert_or_assign(hint, std::move(key), std::move(value));
	}
}

NAMESPACE_END(cereal)
