/// @file
/// @author uentity
/// @date 04.06.2018
/// @brief Declare serialization of base BS types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../objbase.h"

#include "serialize_decl.h"
#include "carray.h"

BSS_FCN_DECL(serialize, blue_sky::objbase)

NAMESPACE_BEGIN(cereal)

template<typename Archive>
auto serialize(Archive& ar, blue_sky::error::box& t) -> void {
	ar(
		make_nvp("code", t.ec),
		make_nvp("domain", t.domain),
		make_nvp("message", t.message)
	);
}

template<typename Archive, typename T, typename E>
auto serialize(Archive& ar, tl::expected<T, E>& t) -> void {
	if constexpr(Archive::is_saving::value) {
		ar(make_nvp("is_expected", t.has_value()));
		if(t.has_value())
			ar(make_nvp("value", *t));
		else
			ar(make_nvp("error", t.error()));
	}
	else {
		bool is_expected;
		ar(make_nvp("is_expected", is_expected));
		if(is_expected)
			ar(make_nvp("value", *t));
		else
			ar(make_nvp("error", t.error()));
	}
}

NAMESPACE_END(cereal)

BSS_FORCE_DYNAMIC_INIT(base_types)
