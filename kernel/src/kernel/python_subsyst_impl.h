/// @file
/// @author uentity
/// @date 23.04.2019
/// @brief BS kernel Python subsystem impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <pybind11/pybind11.h>

#include <bs/objbase.h>

#include <unordered_map>
#include <functional>

#include "python_subsyst.h"

NAMESPACE_BEGIN(blue_sky::kernel::detail)

struct BS_HIDDEN_API python_subsyst_impl : public python_subsyst {
	using adapter_fn = std::function<pybind11::object(sp_obj)>;

	python_subsyst_impl(void* kmod_ptr = nullptr);

	auto py_init_plugin(
		const blue_sky::detail::lib_descriptor& lib, plugin_descriptor& p_descr
	) -> result_or_err<std::string>;

	// construct `error` from any int value -- call after all modules initialized
	auto py_add_error_closure() -> void;

	auto setup_py_kmod(void* kmod_ptr) -> void;

	auto py_kmod() const -> void*;

	auto register_adapter(std::string obj_type_id, adapter_fn f) -> void;

	// pass nullptr to clear
	auto register_default_adapter(adapter_fn f) -> void;

	auto clear_adapters() -> void;

	auto adapted_types() const -> std::vector<std::string>;

	auto adapt(sp_obj source) -> pybind11::object;

	// returns number of cleared instances
	auto drop_adapted_cache(const std::string& obj_id = "") -> std::size_t;

	// access to instance of Python subsystem
	static auto self() -> python_subsyst_impl&;

private:
	void* kmod_;
	// adapters map {obj_type_id -> adapter fcn}
	std::unordered_map<std::string, adapter_fn> adapters_;
	adapter_fn def_adapter_;
	// adapters instances cache
	std::unordered_map<std::string, pybind11::object> acache_;
};

NAMESPACE_END(blue_sky::kernel::detail)
