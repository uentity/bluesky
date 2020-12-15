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
#include "serialize_decl.h"
#include "to_string.h"
#include "base_types.h"
#include "boost_uuid.h"

#include <cereal/types/optional.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <boost/interprocess/streams/vectorstream.hpp>

#include <caf/sec.hpp>
#include <caf/detail/scope_guard.hpp>
#include <caf/detail/stringification_inspector.hpp>

namespace caf {
/*-----------------------------------------------------------------------------
 * Provide `caf::inspect()` overload for types satisfying all conditions below:
 * 1. Serializable by Cereal (check via Cereal traits)
 * 2. Not scalar types (they will conflict with serialization provided by CAF)
 * 3. Not for `caf::stringification_inspector` because we want to provide our own `to_string()`
 * implementation
 *-----------------------------------------------------------------------------*/
// serialize
template<typename Inspector, typename T>
auto inspect(Inspector& f, T& x)
-> std::enable_if_t<
	Inspector::reads_state &&
		!std::is_scalar_v<std::decay_t<T>> &&
		!std::is_same_v<Inspector, caf::detail::stringification_inspector> &&
		!blue_sky::is_archive_inspector_v<Inspector> &&
		cereal::traits::is_output_serializable<T, cereal::PortableBinaryOutputArchive>::value,
	typename Inspector::result_type
> {
	using vostream = boost::interprocess::basic_ovectorstream< std::vector<char> >;
	vostream ss;
	cereal::PortableBinaryOutputArchive A(ss);

	// for BS types run serialization in object's queue
	if constexpr(std::is_base_of_v<::blue_sky::objbase, T>) {
		if(x.apply([&]() -> blue_sky::error {
			A(x);
			return blue_sky::perfect;
		}))
			return sec::state_not_serializable;
	}
	else {
		try {
			A(x);
		}
		catch(...) {
			return sec::state_not_serializable;
		}
	}
	return f.apply_raw(ss.vector().size()*sizeof(char), (void*)ss.vector().data());
}

// deserialize
template<typename Inspector, typename T>
auto inspect(Inspector& f, T& x)
-> std::enable_if_t<
	Inspector::writes_state &&
		!std::is_scalar_v<std::decay_t<T>> &&
		!std::is_same_v<Inspector, caf::detail::stringification_inspector> &&
		!blue_sky::is_archive_inspector_v<Inspector> &&
		cereal::traits::is_input_serializable<T, cereal::PortableBinaryInputArchive>::value,
	typename Inspector::result_type
> {
	using vistream = boost::interprocess::basic_ivectorstream< std::vector<char> >;
	std::vector<char> x_data;
	vistream ss(x_data);
	cereal::PortableBinaryInputArchive A(ss);

	auto g = caf::detail::make_scope_guard([&]() -> error {
		try {
			A(x);
			return none;
		}
		catch(...) {
			return sec::state_not_serializable;
		}
	});
	return f.apply_raw(x_data.size()*sizeof(char), x_data.data());
}

/// to string conversion via JSON archive
template<typename T>
auto to_string(const T& t)
-> std::enable_if_t< sizeof(decltype(::blue_sky::to_string(std::declval<T>()))) != 0, std::string > {
	return ::blue_sky::to_string(t);
}

} // eof caf namespace
