/// @file
/// @author uentity
/// @date 14.01.2019
/// @brief Kernel misc API impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "kimpl.h"
#include "python_subsyst.h"

#include <bs/error.h>
#include <bs/kernel/misc.h>
#include <bs/misc.h>

NAMESPACE_BEGIN(blue_sky::kernel)

auto init() -> error {
	return KIMPL.init();
}

auto shutdown() -> void {
	KIMPL.shutdown();
}

auto unify_serialization() -> void {
	KIMPL.unify_serialization();
}

auto k_descriptor() -> const plugin_descriptor& {
	return KIMPL.kernel_pd();
}

auto k_pymod() -> void* {
	return KIMPL.pysupport()->py_kmod();
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
