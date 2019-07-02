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

NAMESPACE_BEGIN(blue_sky)
/*-----------------------------------------------------------------------------
 *  formatters manipulation API
 *-----------------------------------------------------------------------------*/
using object_saver_fn = std::function<
	error (const objbase& obj, std::ofstream& obj_file, std::string_view fmt_name)
>;

using object_loader_fn = std::function<
	error (objbase& obj, std::ifstream& obj_file, std::string_view fmt_name)
>;

struct object_formatter : std::pair<object_saver_fn, object_loader_fn> {
	using base_t = std::pair<object_saver_fn, object_loader_fn>;

	const std::string name;
	const bool stores_node = false;

	object_formatter(
		std::string fmt_name, object_saver_fn saver, object_loader_fn loader, bool stores_node_ = false
	) : base_t{std::move(saver), std::move(loader)}, name(std::move(fmt_name)), stores_node(stores_node_)
	{}
};

BS_API_PLUGIN auto install_formatter(const type_descriptor& obj_type, object_formatter of) -> bool;
// [NOTE] `fmt_name` as string copy is intentional to avoid side effects
BS_API_PLUGIN auto uninstall_formatter(std::string_view obj_type_id, std::string fmt_name) -> bool;

BS_API_PLUGIN auto formatter_installed(std::string_view obj_type_id, std::string_view fmt_name) -> bool;
BS_API_PLUGIN auto list_installed_formatters(std::string_view obj_type_id) -> std::vector<std::string>;

BS_API_PLUGIN auto get_formatter(std::string_view obj_type_id, std::string_view fmt_name) -> object_formatter*;

NAMESPACE_BEGIN(detail)

inline constexpr auto bin_fmt_name = "bin";
inline constexpr auto json_fmt_name = "json";

NAMESPACE_END(detail)

NAMESPACE_END(blue_sky)
