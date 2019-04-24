/// @file
/// @author uentity
/// @date 23.04.2019
/// @brief BS kernel Python subsystem impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "python_subsyst.h"

NAMESPACE_BEGIN(blue_sky::kernel::detail)

struct BS_HIDDEN_API python_subsyst_impl : public python_subsyst {
	python_subsyst_impl(void* kmod_ptr = nullptr);

	auto py_init_plugin(
		const blue_sky::detail::lib_descriptor& lib, plugin_descriptor& p_descr
	) -> result_or_err<std::string>;

	// construct `error` from any int value -- call after all modules initialized
	auto py_add_error_closure() -> void;

	auto setup_py_kmod(void* kmod_ptr) -> void;

	auto py_kmod() const -> void*;

private:
	void* kmod_;
};

NAMESPACE_END(blue_sky::kernel::detail)
