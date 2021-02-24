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
/*-----------------------------------------------------------------------------
 *  inspector-like tag class for proper dispatching serialization of CAF types
 *  is_inspectable<archive_inspector, CAF type> is always true
 *-----------------------------------------------------------------------------*/
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

/// options for using with Tree FS archives
enum class TFSOpts {
	None = 0,
	// clear dirs when entering 'em
	SaveClearDirs = 1,
	// if not set - node is reconstructed exactly from leafs stored with it's handle link.
	// if set - for any node N all link files found inside N's directory will be loaded into N.
	LoadNodeRecover = 2
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
