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
#include "python_subsyst.h"

#include <spdlog/spdlog.h>

NAMESPACE_BEGIN(blue_sky::kernel)

auto init() -> error {
	using InitState = kimpl::InitState;

	// do initialization only once from non-initialized state
	auto expected_state = InitState::NonInitialized;
	if(KIMPL.init_state_.compare_exchange_strong(expected_state, InitState::Initialized)) {
		// if init wasn't finished - return kernel to non-initialized status
		auto init_ok = false;
		auto finally = scope_guard{ [&]{ if(!init_ok) KIMPL.init_state_ = InitState::NonInitialized; } };

		// configure kernel
		KIMPL.configure();
		// switch to mt logs
		KIMPL.toggle_mt_logs(true);
		// init kernel radio subsystem
		auto er = KIMPL.init_radio();
		init_ok = er.ok();
		return er;
	}
	return perfect;
}

auto shutdown() -> void {
	using InitState = kimpl::InitState;

	// shut down if not already Down
	auto expected_state = InitState::Initialized;
	if(KIMPL.init_state_.compare_exchange_strong(expected_state, InitState::Down)) {
		// turn off radio subsystem
		KIMPL.shutdown_radio();
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
	return KIMPL.pysupport_->py_kmod();
}

auto last_error() -> std::string {
	return last_system_message();
}

auto str_key_storage(const std::string& key) -> str_any_array& {
	auto& kimpl = KIMPL;
	auto solo = std::lock_guard{ kimpl.sync_storage_ };
	return kimpl.str_key_storage(key);
}

auto idx_key_storage(const std::string& key) -> idx_any_array& {
	auto& kimpl = KIMPL;
	auto solo = std::lock_guard{ kimpl.sync_storage_ };
	return kimpl.idx_key_storage(key);
}

NAMESPACE_END(blue_sky::kernel)
