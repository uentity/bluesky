/// @file
/// @author uentity
/// @date 21.12.2018
/// @brief BS kernel plugins API
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../common.h"
#include "../plugin_descriptor.h"
#include "type_tuple.h"

NAMESPACE_BEGIN(blue_sky::kernel::plugins)

using plugins_enum = std::vector<const plugin_descriptor*>;
using types_enum = std::vector<tfactory::type_tuple>;

/// Direct register plugin if shared lib is already loaded
BS_API auto register_plugin(const plugin_descriptor* pd) -> bool;
/// Dynamically loads particular plugin
BS_API auto load_plugin(const std::string& fname, bool init_py_subsyst) -> int;
/// Load blue-sky plugins method
BS_API auto load_plugins(void* py_root_module = nullptr) -> int;
/// Unloads plugin
BS_API auto unload_plugin(const plugin_descriptor& pd) -> void;
/// Unload blue-sky plugins method
BS_API auto unload_plugins() -> void;

BS_API auto loaded_plugins() -> plugins_enum;

//! \brief registered type infos of objects
BS_API auto registered_types() -> types_enum;
//! \brief types of plugin (by plugin descriptor)
BS_API auto plugin_types(const plugin_descriptor& pd) -> types_enum;
BS_API auto plugin_types(const std::string& plugin_name) -> types_enum;

NAMESPACE_END(blue_sky::kernel::plugins)
