/// @file
/// @author uentity
/// @date 14.01.2019
/// @brief Kernel misc API impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include <bs/kernel/misc.h>
#include <bs/misc.h>
#include "kimpl.h"

#include <spdlog/spdlog.h>

NAMESPACE_BEGIN(blue_sky::kernel)

auto init() -> void {
	using InitState = kimpl::InitState;

	// do initialization only once from non-initialized state
	auto expected_state = InitState::NonInitialized;
	if(KIMPL.init_state_.compare_exchange_strong(expected_state, InitState::Initialized)) {
		// configure kernel
		KIMPL.configure();
		// switch to mt logs
		KIMPL.toggle_mt_logs(true);
		// init actor system
		auto& actor_sys = KIMPL.actor_sys_;
		if(!actor_sys) {
			actor_sys = std::make_unique<caf::actor_system>(KIMPL.actor_cfg_);
			if(!actor_sys)
				throw error("Can't create CAF actor_system!");
		}
	}
}

auto shutdown() -> void {
	using InitState = kimpl::InitState;

	// shut down if not already Down
	if(KIMPL.init_state_.exchange(InitState::Down) != InitState::Down) {
		// destroy actor system
		if(KIMPL.actor_sys_) {
			KIMPL.actor_sys_.release();
		}
		// shutdown mt logs
		KIMPL.toggle_mt_logs(false);
		spdlog::shutdown();
	}
}

auto unify_serialization() -> void {
	KIMPL.unify_serialization();
}

auto k_descriptor() -> const plugin_descriptor& {
	return KIMPL.kernel_pd();
}

auto k_pymod() -> void* {
	return KIMPL.self_pymod();
}

auto last_error() -> std::string {
	return last_system_message();
}

auto pert_str_any_array(const type_descriptor& master) -> str_any_array& {
	return KIMPL.pert_str_any_array(master);
}

auto pert_idx_any_array(const type_descriptor& master) -> idx_any_array& {
	return KIMPL.pert_idx_any_array(master);
}

NAMESPACE_END(blue_sky::kernel)
