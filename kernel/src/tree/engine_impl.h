/// @file
/// @author uentity
/// @date 01.07.2020
/// @brief Describes engine::impl class
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/actor_common.h>
#include <bs/tree/engine.h>
#include <bs/detail/sharded_mutex.h>

#include <caf/detail/shared_spinlock.hpp>

#include <unordered_map>

NAMESPACE_BEGIN(blue_sky::tree)

using engine_impl_mutex = caf::detail::shared_spinlock;

/// tree element must inherit impl class from this one
struct engine::impl {
	using sp_engine_impl = std::shared_ptr<impl>;
	using sp_scoped_actor = std::shared_ptr<caf::scoped_actor>;

	/// return engine's type ID
	virtual auto type_id() const -> std::string_view = 0;

	auto factor(const void* L) -> sp_scoped_actor; 
	auto release_factor(const void* L) -> void;
	auto release_factors() -> void;
	
private:
	// requesters pool { link addr -> `scoped_actor` instance }
	using rpool_t = std::unordered_map<const void*, sp_scoped_actor>;
	rpool_t rpool_;

	blue_sky::detail::sharded_mutex<engine_impl_mutex> guard_;
};

NAMESPACE_END(blue_sky::tree)
