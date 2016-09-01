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
#include <spdlog/logger.h>
#include <unordered_map>

namespace blue_sky { namespace detail {

struct BS_HIDDEN_API kernel_logging_subsyst {
	std::unordered_map< std::string, std::shared_ptr< spdlog::logger > > logs_;

	kernel_logging_subsyst();

	spdlog::logger& get_log(const char* log_name = "out");
};
	
}} /* namespace blue_sky::detail */

