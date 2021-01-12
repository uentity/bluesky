/// @file
/// @author uentity
/// @date 14.04.2020
/// @brief Serialize CAF inspectable types to Cereal archives
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/kernel/radio.h>
#include "serialize_decl.h"
#include "carray.h"

#include <caf/binary_serializer.hpp>
#include <caf/binary_deserializer.hpp>

/*-----------------------------------------------------------------------------
 *  overload of `serialize()` for CAF types
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN(cereal)

template<typename Archive, typename T>
auto serialize(Archive& ar, T& t)
-> std::enable_if_t< caf::detail::is_inspectable<blue_sky::archive_inspector<Archive>, std::decay_t<T>>::value, void > {
	using namespace cereal;

	std::vector<char> buf;
	if constexpr(Archive::is_saving::value) {
		auto archvile = caf::binary_serializer{ blue_sky::kernel::radio::system(), buf };
		inspect(archvile, t);

		// [NOTE] use write_pos() because `buf` size can be larger than actual data
		save_carray(ar, buf.data(), archvile.write_pos());
	}
	else {
		load_carray(ar, buf.data(), [&](auto*, auto sz) {
			buf.resize(sz);
			return buf.data();
		});

		auto archvile = caf::binary_deserializer{ blue_sky::kernel::radio::system(), buf };
		inspect(archvile, t);
	}
}

NAMESPACE_END(cereal)
