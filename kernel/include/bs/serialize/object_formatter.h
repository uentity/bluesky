/// @file
/// @author uentity
/// @date 30.06.2019
/// @brief Object formatters API used in Tree FS archive
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../common.h"
#include "../error.h"

NAMESPACE_BEGIN(blue_sky)
/*-----------------------------------------------------------------------------
 *  formatters manipulation API
 *-----------------------------------------------------------------------------*/
using object_saver_fn = std::function<
	error (const objbase& obj, std::string obj_fname, std::string_view fmt_name)
>;

using object_loader_fn = std::function<
	error (objbase& obj, std::string obj_fname, std::string_view fmt_name)
>;

struct BS_API object_formatter : private std::pair<object_saver_fn, object_loader_fn> {
	using base_t = std::pair<object_saver_fn, object_loader_fn>;

	const std::string name;
	const bool stores_node = false;

	object_formatter(
		std::string fmt_name, object_saver_fn saver, object_loader_fn loader, bool stores_node = false
	);

	auto save(const objbase& obj, std::string obj_fname) const -> error;
	auto load(objbase& obj, std::string obj_fname) const -> error;

	template<typename, typename> friend struct formatter_tools;
};

BS_API auto install_formatter(const type_descriptor& obj_type, object_formatter of) -> bool;
// [NOTE] `fmt_name` as string copy is intentional to avoid side effects
BS_API auto uninstall_formatter(std::string_view obj_type_id, std::string fmt_name) -> bool;

BS_API auto formatter_installed(std::string_view obj_type_id, std::string_view fmt_name) -> bool;
BS_API auto list_installed_formatters(std::string_view obj_type_id) -> std::vector<std::string>;

BS_API auto get_formatter(std::string_view obj_type_id, std::string_view fmt_name) -> object_formatter*;
BS_API auto get_obj_formatter(const objbase* obj) -> object_formatter*;

NAMESPACE_BEGIN(detail)

inline constexpr auto bin_fmt_name = "bin";
inline constexpr auto json_fmt_name = "json";

NAMESPACE_END(detail)

NAMESPACE_END(blue_sky)
