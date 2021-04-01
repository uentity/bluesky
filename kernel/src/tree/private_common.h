/// @date 19.10.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/defaults.h>
#include <bs/detail/enumops.h>
#include <bs/tree/common.h>

#include <caf/typed_actor.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
namespace bs_detail = blue_sky::detail;

using defaults::tree::nil_uid;
using defaults::tree::nil_oid;
inline const auto nil_otid = blue_sky::defaults::nil_type_name;

/// link erase options
enum class EraseOpts { Normal = 0, Silent = 1 };

enum class ReqOpts : std::uint32_t {
	WaitIfBusy = 0, ErrorIfBusy = 1, ErrorIfNOK = 2, DirectInvoke = 4,
	HasDataCache = 8, Uniform = 16, Detached = 512, TrackWorkers = 1024
};

/// messages processed by any engine-derived class
template<typename Engine>
using engine_actor_type = caf::typed_actor<
	// obtain engine impl
	caf::replies_to<a_impl>::with<engine::sp_engine_impl>,
	// clone engine impl
	typename caf::replies_to<a_clone, a_impl, bool /* deep */>
	::template with<std::shared_ptr<typename Engine::engine_impl>>,
	// add listener actor to home group
	caf::replies_to<a_subscribe, caf::actor /* ev listener */>::with<std::uint64_t>
>;

/// common interface of engine home group
using engine_home_actor_type = caf::typed_actor<
	// sent by home owner on exit
	caf::reacts_to<a_bye>
>;

using ev_listener_actor_type = caf::typed_actor<
	// returns ev listener actor ID
	caf::replies_to<a_hi, caf::group /* source home */>::with<std::uint64_t>,
	// ceases to exit
	caf::reacts_to<a_bye>
>;

NAMESPACE_END(blue_sky::tree)

BS_ALLOW_ENUMOPS(tree::ReqOpts)
