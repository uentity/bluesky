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
#include <bs/tree/link.h>
#include "python_subsyst.h"
#include "radio_subsyst.h"

#include <caf/scoped_actor.hpp>

#include <unordered_map>
#include <functional>
#include <mutex>

NAMESPACE_BEGIN(blue_sky::kernel::detail)

using pyqueue_actor_type = caf::typed_actor<
	caf::replies_to<simple_transaction>::with<error::box>
>::extend_with<khome_actor_type>;

struct BS_HIDDEN_API python_subsyst_impl : public python_subsyst {
	using adapter_fn = std::function<pybind11::object(sp_obj)>;

	python_subsyst_impl(void* kmod_ptr = nullptr);
	~python_subsyst_impl();

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

	auto adapt(sp_obj source, const tree::link& L) -> pybind11::object;

	auto get_cached_adapter(const sp_obj& source) const -> pybind11::object;
	// returns number of cleared instances
	auto drop_adapted_cache(const sp_obj& source = nullptr) -> std::size_t;

	// post transaction into Python's queue
	auto enqueue(simple_transaction tr) -> error;
	auto enqueue(launch_async_t, simple_transaction tr) -> void;

	// access to instance of Python subsystem
	static auto self() -> python_subsyst_impl&;

private:
	void* kmod_;
	// adapters map {obj_type_id -> adapter fcn}
	std::unordered_map<std::string, adapter_fn> adapters_;
	adapter_fn def_adapter_;

	// cache value = { adapter instance, ref_counter }
	// when `ref_counter` reaches zero, cache value is dropped
	// ref counter counts links that point to source object
	using acache_value = std::pair<pybind11::object, size_t>;
	// adapters cache { object ptr, adapter }
	std::unordered_map<const objbase*, acache_value> acache_;
	// used to resolve link ID -> object pointer when link is erased
	// by keeping this map we can omit (expensive) call to `link::data()`
	// [NOTE] link ID is stored as string for convinience
	std::unordered_map<std::string, const objbase*> lnk2obj_;

	mutable std::mutex guard_;

	// queue
	pyqueue_actor_type queue_;
	union { caf::scoped_actor queue_factor_; };
};

NAMESPACE_END(blue_sky::kernel::detail)
