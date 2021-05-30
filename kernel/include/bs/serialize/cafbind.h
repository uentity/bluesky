/// @file
/// @author uentity
/// @date 13.08.2018
/// @brief Allow CAF messages to be serialized using Cereal-based code
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../error.h"
#include "../type_caf_id.h"
#include "serialize_decl.h"
#include "to_string.h"
#include "base_types.h"
#include "boost_uuid.h"

#include <cereal/types/optional.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <boost/interprocess/streams/vectorstream.hpp>

#include <caf/allowed_unsafe_message_type.hpp>
#include <caf/inspector_access_type.hpp>
#include <caf/detail/stringification_inspector.hpp>

NAMESPACE_BEGIN(blue_sky::detail)

// this resembles logic of `caf::inspect_access_type()` with removed paths
// that check for `inspector_access` and `inspect` overloads
// return: false if CAF provides machinery to serialize `T`, true otherwise
template <class Inspector, class T>
constexpr auto allow_bs_inspect() -> bool {
	using namespace caf;
	using namespace caf::detail;

	if constexpr(
		is_allowed_unsafe_message_type_v<T>
		|| std::is_array<T>::value
		|| caf::detail::is_builtin_inspector_type<T, Inspector::is_loading>::value
		|| has_builtin_inspect<Inspector, T>::value
		|| std::is_empty<T>::value
		|| is_stl_tuple_type_v<T>
		|| is_map_like_v<T>
		|| is_list_like_v<T>
	)
		return false;
	else
		return true;
}

NAMESPACE_END(blue_sky::detail)

NAMESPACE_BEGIN(blue_sky)

/// one of switches that enables BS generic `inspect()` overload
template<typename Inspector, typename T>
constexpr auto enable_bs_inspect() -> bool {
	constexpr bool res =
		!std::is_same_v<Inspector, caf::detail::stringification_inspector>
		&& !is_archive_inspector<Inspector>
		&& detail::allow_bs_inspect<Inspector, T>()
	;

	if constexpr(Inspector::is_loading)
		return res && cereal::traits::is_input_serializable<T, cereal::PortableBinaryInputArchive>::value;
	else
		return res && cereal::traits::is_output_serializable<T, cereal::PortableBinaryOutputArchive>::value;
}

NAMESPACE_END(blue_sky)

NAMESPACE_BEGIN(caf)
/*-----------------------------------------------------------------------------
 * Provide `caf::inspect()` overload for types satisfying all conditions below:
 * 1. Serializable by Cereal (check via Cereal traits)
 * 2. Not scalar types (they will conflict with serialization provided by CAF)
 * 3. Not for `caf::stringification_inspector` because we want to provide our own `to_string()`
 * implementation
 *-----------------------------------------------------------------------------*/
// -------- save
template<
	typename Inspector, typename T,
	typename = std::enable_if_t<blue_sky::enable_bs_inspect<Inspector, T>()>
>
auto inspect(Inspector& f, T& x) -> bool {
	// ==== save path
	if constexpr(!Inspector::is_loading) {
		using vostream = boost::interprocess::basic_ovectorstream< std::vector<char> >;
		std::vector<char> x_data;
		vostream ss(x_data);
		cereal::PortableBinaryOutputArchive A(ss);

		// for BS types run serialization in object's queue
		if constexpr(std::is_base_of_v<::blue_sky::objbase, T>) {
			if(x.apply([&]() -> blue_sky::error {
				A(x);
				return blue_sky::perfect;
			}))
				return false;
		}
		else {
			try {
				A(x);
			}
			catch(...) {
				return false;
			}
		}
		return f.apply(ss.vector());
	}
	// ==== load path
	else {
		using vistream = boost::interprocess::basic_ivectorstream< std::vector<char> >;
		std::vector<char> x_data;

		if(f.apply(x_data)) {
			try {
				vistream ss(x_data);
				cereal::PortableBinaryInputArchive A(ss);
				A(x);
				return true;
			}
			catch(...) {
				return false;
			}
		};
		return false;
	}
}

// temp use unsafe conversions from enum -> underlying type (mute CAF warnings)
template<typename Inspector, typename T>
auto inspect(Inspector& f, T&& x)
-> std::enable_if_t<
	std::is_enum_v<blue_sky::meta::remove_cvref_t<T>>,
	bool
> {
	using E = blue_sky::meta::remove_cvref_t<T>;
	using U = std::underlying_type_t<E>;
	if constexpr(Inspector::is_loading) {
		auto tmp = U{0};
		if(f.value(tmp)) {
			x = static_cast<E>(tmp);
			return true;
		}
		else return false;
	}
	else {
		return f.value(static_cast<U>(x));
	}
}

/// to string conversion via JSON archive
template<typename T, typename = decltype( blue_sky::to_string(std::declval<T>()) )>
auto to_string(const T& t) {
	return blue_sky::to_string(t);
}

NAMESPACE_END(caf)
