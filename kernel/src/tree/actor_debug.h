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

#include <caf/actor_ostream.hpp>

#ifndef DEBUG_ACTOR
	#define DEBUG_ACTOR 0
#endif

static inline constexpr auto adbg(...) { return blue_sky::log::D(); }

#if DEBUG_ACTOR == 1
	#include "engine_actor.h"

	NAMESPACE_BEGIN(blue_sky::tree)

	auto adbg_impl(caf::actor_ostream, const link_impl&) -> caf::actor_ostream;
	auto adbg_impl(caf::actor_ostream, const node_impl&) -> caf::actor_ostream;

	auto adbg_impl(engine_actor_base*) -> caf::actor_ostream;
	static auto adbg(engine_actor_base* A) { return adbg_impl(A); }

	template<typename Item>
	static auto adbg(engine_actor<Item>* A) {
		return adbg_impl(caf::aout(A), A->impl);
	}

	static auto adbg(caf::event_based_actor* A) {
		return caf::actor_ostream(A);
	}

	NAMESPACE_END(blue_sky::tree)
#endif
