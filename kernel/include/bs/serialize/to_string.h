/// @file
/// @author uentity
/// @date 20.06.2019
/// @brief Fallback `to_string()` implementation to JSON format
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <sstream>
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>

namespace blue_sky {

template<typename T>
auto to_string(const T& t)
-> std::enable_if_t<
	!std::is_scalar_v<std::decay_t<T>> &&
		cereal::traits::is_output_serializable<std::decay_t<T>, cereal::JSONOutputArchive>::value,
	std::string
> {
	std::ostringstream ss;
	{
		cereal::JSONOutputArchive A(ss);
		A(t);
	}
	return ss.str();
}

} // eof blue_sky
