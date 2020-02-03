/// @file
/// @author uentity
/// @date 03.02.2020
/// @brief Debug prints support
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/log.h>

#ifndef DEBUG_ACTOR
	#define DEBUG_ACTOR 0
#endif

#if DEBUG_ACTOR == 1
#include <caf/actor_ostream.hpp>
#endif

NAMESPACE_BEGIN(blue_sky::tree)

#if DEBUG_ACTOR == 0

template<typename Actor, typename... Ts>
static constexpr auto adbg(Actor*, Ts&&...) {
	return blue_sky::log::D();
}

#endif

NAMESPACE_END(blue_sky::tree)
