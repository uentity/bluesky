/// @file
/// @author uentity
/// @date 19.11.2018
/// @brief BS kernel config subsystem API
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/common.h>
#include <caf/actor_system_config.hpp>

#include <filesystem>

#define KCONFIG ::blue_sky::singleton<::blue_sky::kernel::detail::config_subsyst>::Instance()

NAMESPACE_BEGIN(blue_sky::kernel::detail)

struct BS_HIDDEN_API config_subsyst {
	using string_list = std::vector<std::string>;

	config_subsyst();

	/// impl is taken from CAF sources
	/// parse config options from arguments and ini file content
	/// if `force` is true, force config files reparse
	auto configure(string_list args = {}, std::string ini_fname = "", bool force = false) -> void;

	auto clear_confdata() -> void;

	static auto is_configured() -> bool;

	// predefined config options that can be parsed from CLI or config file
	caf::config_option_set confopt_;
	// config values storage: map from string key -> any parsed value
	caf::settings confdata_;
	// kernel's actor system & config
	caf::actor_system_config actor_cfg_;

private:
	// paths of possible config file location
	std::vector<std::filesystem::path> conf_path_;
	// flag indicating if kernel was configured at least once
	static bool kernel_configured;
};

NAMESPACE_END(blue_sky::kernel::detail)
