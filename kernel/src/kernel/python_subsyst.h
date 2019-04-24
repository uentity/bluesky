/// @file
/// @author uentity
/// @date 23.04.2019
/// @brief BS kernel Python subsystem iface
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/common.h>
#include <bs/error.h>
#include <bs/detail/lib_descriptor.h>

NAMESPACE_BEGIN(blue_sky::kernel::detail)

struct python_subsyst {
	virtual auto py_init_plugin(
		const blue_sky::detail::lib_descriptor& lib, plugin_descriptor& p_descr
	) -> result_or_err<std::string> = 0;

	// construct `error` from any int value -- call after all modules initialized
	virtual auto py_add_error_closure() -> void = 0;

	// return & set kernel's pybind11::module address
	virtual auto py_kmod() const -> void* = 0;
	virtual auto setup_py_kmod(void* kmod_ptr) -> void = 0;
};

NAMESPACE_END(blue_sky::kernel::detail)
