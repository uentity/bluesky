/// @file
/// @author uentity
/// @date 20.12.2018
/// @brief BS kernel configure API
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../common.h"

#include <caf/config_value.hpp>
#include <caf/config_option_adder.hpp>

NAMESPACE_BEGIN(blue_sky::kernel::config)

/// configure kernel
BS_API auto configure(
	std::vector<std::string> args = {}, std::string ini_fname = "", bool force = false
) -> const caf::config_value_map&;

BS_API auto is_configured() -> bool;

/// access to kernel's config variables
BS_API auto config() -> const caf::config_value_map&;

/// allows to add custom options to specified config section
BS_API auto config_section(std::string_view name) -> caf::config_option_adder;
/// access global CAF actor_system_config
BS_API auto actor_config() -> caf::actor_system_config&;
/// ... and to actors system
BS_API auto actor_system() -> caf::actor_system&;

NAMESPACE_END(blue_sky::kernel::config)
