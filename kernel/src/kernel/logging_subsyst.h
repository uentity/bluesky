/// @file
/// @author uentity
/// @date 24.08.2016
/// @brief Create and register BlueSky logs
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/common.h>

#include <spdlog/common.h>
#include <spdlog/logger.h>

NAMESPACE_BEGIN(blue_sky::kernel::detail)

struct BS_HIDDEN_API logging_subsyst {
	logging_subsyst();

	static auto null_logger() -> const std::shared_ptr<spdlog::logger>&;

	static auto toggle_async(bool turn_on) -> void;

	// if `logger_name` isn't specified, add to all existing logs
	static auto add_custom_sink(spdlog::sink_ptr sink, const std::string& logger_name = {}) -> void;

	static auto remove_custom_sink(spdlog::sink_ptr sink, const std::string& logger_name = {}) -> void;

	static auto shutdown() -> void;
};

NAMESPACE_END(blue_sky::kernel::detail)
