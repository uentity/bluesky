/// @date 19.08.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../meta.h"
#include <memory>

namespace blue_sky::meta {

template<typename T>
struct is_shared_ptr : std::false_type {};

template<typename T>
struct is_shared_ptr< std::shared_ptr<T> > : std::true_type {
	using pointee = T;
};

template<typename T>
inline constexpr auto is_shared_ptr_v = is_shared_ptr<remove_cvref_t<T>>::value;

} // eof blue_sky::meta
