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

#include <caf/allowed_unsafe_message_type.hpp>
#include <caf/inspector_access_type.hpp>
#include <caf/binary_serializer.hpp>
#include <caf/binary_deserializer.hpp>

NAMESPACE_BEGIN(blue_sky)

/// test if `T` can be inspected solely by CAF excluding BS `inspect()` overload
template<typename Archive, typename T>
inline constexpr bool is_caf_builtin_inspectable = !std::is_same_v<
	decltype(caf::inspect_access_type<archive_inspector<Archive::is_loading::value>, T>()),
	caf::inspector_access_type::none
>;

NAMESPACE_END(blue_sky)

/*-----------------------------------------------------------------------------
 *  overload of `serialize()` for CAF types
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN(cereal)

template<typename Archive, typename T>
auto serialize(Archive& ar, T& t)
-> std::enable_if_t<blue_sky::is_caf_builtin_inspectable<Archive, T>> {
	using namespace cereal;

	caf::byte_buffer buf;
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
