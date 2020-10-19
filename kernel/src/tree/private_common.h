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

enum class ReqOpts {
	WaitIfBusy = 0, ErrorIfBusy = 1, ErrorIfNOK = 2, Detached = 4, DirectInvoke = 8,
	HasDataCache = 16, Uniform = 32
};

/// messages processed by any engine-derived class
template<typename Engine>
using engine_actor_type = caf::typed_actor<
	// obtain engine impl
	caf::replies_to<a_impl>::with<std::shared_ptr<typename Engine::engine_impl>>,
	// run transaction in engine's queue
	caf::replies_to<a_apply, simple_transaction>::with<error::box>,
	typename caf::replies_to<a_apply, transaction_t<error, typename Engine::bare_type>>
	::template with<error::box>,
	// add listener actor to home group
	caf::replies_to<a_subscribe, caf::actor /* ev listener */>::with<std::uint64_t>
>;

using ev_listener_actor_type = caf::typed_actor<
	// returns ev listener actor ID
	caf::replies_to<a_hi, caf::group /* source home */>::with<std::uint64_t>,
	// ceases to exit
	caf::reacts_to<a_bye>
>;

NAMESPACE_END(blue_sky::tree)
