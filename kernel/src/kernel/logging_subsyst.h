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

NAMESPACE_BEGIN(blue_sky::kernel::detail)

struct BS_HIDDEN_API logging_subsyst {

	logging_subsyst();

	static auto toggle_mt_logs(bool turn_on) -> void;
};

NAMESPACE_END(blue_sky::kernel::detail)
