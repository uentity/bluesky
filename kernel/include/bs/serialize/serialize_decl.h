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
#include "../detail/enumops.h"

#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>

NAMESPACE_BEGIN(blue_sky)

 /// inspector-like tag class for proper dispatching serialization of builtin CAF types
template<bool IsLoading>
struct archive_inspector {
	static constexpr bool is_loading = IsLoading;

	template<typename... Ts>
	auto operator()(Ts&&...) { return true; }
};

NAMESPACE_BEGIN(detail)

template<typename T>
struct is_archive_inspector : std::false_type {};

template<bool IsLoading>
struct is_archive_inspector<archive_inspector<IsLoading>> : std::true_type {};

NAMESPACE_END(detail)

/// test if inspector is `archive_inspector`
template<typename T>
inline constexpr bool is_archive_inspector = detail::is_archive_inspector<T>::value;

/// options for using with Tree FS archives
enum class TFSOpts : std::uint8_t {
	None = 0,
	// clear dirs when entering 'em (on saving)
	ClearDirs = 1,
	// force clear objects dir (on saving)
	ClearObjectsDir = 2
};

// forward declare Tree FS archives
class tree_fs_input;
class tree_fs_output;

NAMESPACE_END(blue_sky)

BS_ALLOW_ENUMOPS(TFSOpts)

/*-----------------------------------------------------------------------------
 *  Define specific version tag for archives used
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN(cereal)
// forward declare text Cereal archives
class JSONInputArchive;
class JSONOutputArchive;

NAMESPACE_BEGIN(traits)

template<typename Archive>
struct class_version_tag<Archive, std::enable_if_t<
	std::is_same_v<Archive, blue_sky::tree_fs_input>
	|| std::is_same_v<Archive, blue_sky::tree_fs_output>
	|| std::is_same_v<Archive, JSONInputArchive>
	|| std::is_same_v<Archive, JSONOutputArchive>
>> {
	static constexpr auto value = "bs_class_version";
};

NAMESPACE_END(traits)
NAMESPACE_END(cereal)
