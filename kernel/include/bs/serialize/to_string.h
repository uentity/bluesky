/// @author uentity
/// @date 20.06.2019
/// @brief Fallback `to_string()` implementation to JSON format
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "serialize_decl.h"
#include <cereal/archives/json.hpp>

#include <caf/deep_to_string.hpp>

#include <sstream>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(detail)

template<typename T>
inline constexpr auto has_caf_builtin_deep2string(int)
-> decltype(deep_to_string(std::declval<const T&>()), bool{}) { return true; }

template<typename T>
inline constexpr auto has_caf_builtin_deep2string(...) { return false; }

NAMESPACE_END(detail)

template<typename T>
auto to_string(const T& t)
-> std::enable_if_t<
	!detail::has_caf_builtin_deep2string<T>(0)
	&& cereal::traits::is_output_serializable<T, cereal::JSONOutputArchive>::value,
	std::string
> {
	std::ostringstream ss;
	{
		cereal::JSONOutputArchive A(ss);
		A(t);
	}
	return ss.str();
}

NAMESPACE_END(blue_sky)
