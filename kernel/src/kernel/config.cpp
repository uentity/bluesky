/// @file
/// @author uentity
/// @date 14.01.2019
/// @brief Kernel config API impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel/config.h>
#include "kimpl.h"
#include "config_subsyst.h"

NAMESPACE_BEGIN(blue_sky::kernel::config)

auto configure(
	std::vector<std::string> args, std::string ini_fname, bool force
) -> const caf::settings& {
	KIMPL.get_config()->configure(std::move(args), ini_fname, force);
	return KIMPL.get_config()->confdata_;
}

auto is_configured() -> bool {
	return detail::config_subsyst::is_configured();
}

auto config() -> const caf::settings& {
	return KIMPL.get_config()->confdata_;
}

auto config_section(std::string_view name) -> caf::config_option_adder {
	return caf::config_option_adder(KIMPL.get_config()->confopt_, name);
}

auto actor_config() -> caf::actor_system_config& {
	return KIMPL.get_config()->actor_cfg_;
}

NAMESPACE_END(blue_sky::kernel::config)
