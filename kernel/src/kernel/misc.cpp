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
#include <bs/uuid.h>

#include <boost/uuid/string_generator.hpp>

NAMESPACE_BEGIN(blue_sky)

auto gen_uuid() -> uuid {
	return KIMPL.gen_uuid();
}

auto to_uuid(std::string_view s) noexcept -> result_or_err<uuid> {
	auto res = result_or_err<uuid>{};
	auto er = error::eval_safe([&] {
		res = boost::uuids::string_generator{}(s.begin(), s.end());
	});
	return er.ok() ? res : tl::make_unexpected(std::move(er));
}

auto to_uuid(unsafe_t, std::string_view s) -> uuid {
	return boost::uuids::string_generator{}(s.begin(), s.end());
}

NAMESPACE_BEGIN(kernel)
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
	return KIMPL.str_key_storage(key);
}

auto idx_key_storage(const std::string& key) -> idx_any_array& {
	return KIMPL.idx_key_storage(key);
}

NAMESPACE_END(kernel)
NAMESPACE_END(blue_sky)
