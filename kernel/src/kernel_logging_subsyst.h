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
#include <bs/log.h>
#include <spdlog/logger.h>

namespace blue_sky { namespace detail {

struct BS_HIDDEN_API kernel_logging_subsyst {

	kernel_logging_subsyst() {
		// ensure that log globals are created before kernel
		// that means log will be alive as long as kernel alive
		const spdlog::logger* const init_logs[] = {
			&log::get_logger("out"), &log::get_logger("err")
		};
		(void)init_logs;
	}

	static auto toggle_mt_logs(bool turn_on) -> void;
};

}} /* namespace blue_sky::detail */

