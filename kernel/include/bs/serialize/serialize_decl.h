/// @file
/// @author uentity
/// @date 20.06.2019
/// @brief Include this into header with declarations of your type serialization fucntions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "atomizer.h"
#include "macro.h"
#include "make_base_class.h"

#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>

/*-----------------------------------------------------------------------------
 *  inspector-like tag class for proper dispatching serialization of CAF types
 *  is_inspectable<archive_inspector, CAF type> is always true
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN(blue_sky)

template<typename Archive>
struct archive_inspector {
	using result_type = void;

	template<typename... Ts>
	auto operator()(Ts&&...) {}
};

template<typename T>
struct is_archive_inspector : std::false_type {};

template<typename Archive>
struct is_archive_inspector< archive_inspector<Archive> > : std::true_type {};

template<typename T>
inline constexpr auto is_archive_inspector_v = is_archive_inspector<T>::value;

NAMESPACE_END(blue_sky)

