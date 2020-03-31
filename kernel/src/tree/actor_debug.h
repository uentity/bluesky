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
#include <bs/tree/common.h>

#include <caf/actor_ostream.hpp>

#ifndef DEBUG_ACTOR
	#define DEBUG_ACTOR 0
#endif

NAMESPACE_BEGIN(blue_sky::tree)

#if DEBUG_ACTOR == 0

template<typename Actor, typename... Ts>
static constexpr auto adbg(Actor*, Ts&&...) {
	return blue_sky::log::D();
}

#else

static auto adbg_impl(link_actor*) -> caf::actor_ostream;
static auto adbg_impl(node_actor*) -> caf::actor_ostream;

template<typename Actor>
static auto adbg(Actor* A) { return adbg_impl(A); }

#endif

NAMESPACE_END(blue_sky::tree)
