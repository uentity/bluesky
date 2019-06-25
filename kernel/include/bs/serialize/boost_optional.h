/// @file
/// @author uentity
/// @date 15.05.2019
/// @brief boost::optional serialization support
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <cereal/cereal.hpp>
#include <boost/optional.hpp>

namespace cereal {

//! Saving for std::optional
template <class Archive, typename T> inline
void CEREAL_SAVE_FUNCTION_NAME(Archive& ar, const boost::optional<T>& optional) {
	if(!optional) {
		ar(CEREAL_NVP_("nullopt", true));
	} else {
		ar(CEREAL_NVP_("nullopt", false),
				CEREAL_NVP_("data", *optional));
	}
}

//! Loading for std::optional
template <class Archive, typename T> inline
void CEREAL_LOAD_FUNCTION_NAME(Archive& ar, boost::optional<T>& optional) {
	bool nullopt;
	ar(CEREAL_NVP_("nullopt", nullopt));

	if (nullopt) {
		optional = boost::optional<T>();
	} else {
		T value;
		ar(CEREAL_NVP_("data", value));
		optional = std::move(value);
	}
}

} // namespace cereal

