/// @file
/// @author uentity
/// @date 24.07.2019
/// @brief BS kernel radio subsystem API
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/common.h>
#include <bs/tree/link.h>

#include <caf/actor_addr.hpp>
#include <caf/actor.hpp>
#include <caf/typed_actor.hpp>
#include <caf/actor_system.hpp>
#include <caf/group.hpp>
#include <caf/scoped_actor.hpp>
#include <caf/detail/shared_spinlock.hpp>

#include <optional>
#include <set>
#include <unordered_set>

#define KRADIO ::blue_sky::singleton<::blue_sky::kernel::detail::radio_subsyst>::Instance()

NAMESPACE_BEGIN(blue_sky::kernel::detail)
// kernel's queue interface
using kqueue_actor_type = caf::typed_actor<
	caf::replies_to<transaction>::with<tr_result::box>
>;

struct BS_HIDDEN_API radio_subsyst {
	// store links that will be visible to the world
	std::set< tree::link, std::less<> > publinks;

	radio_subsyst();

	auto init() -> error;
	auto shutdown() -> void;

	inline auto system() -> caf::actor_system& {
		return (this->*get_actor_sys_)();
	}

	// collect event-based actor adresses that must exit with kernel
	auto register_citizen(caf::actor_addr citizen) -> void;
	inline auto register_citizen(const caf::abstract_actor* citizen) {
		register_citizen(citizen->address());
	}

	// returns whether zitizen were found and removed from registry
	auto release_citizen(const caf::actor_addr& citizen) -> void;
	inline auto release_citizen(const caf::abstract_actor* citizen) {
		release_citizen(citizen->address());
	}

	// send exit message to all citizens & wait until they exit
	auto kick_citizens() -> void;

	// post transaction into kernel's queue
	auto queue_actor() -> kqueue_actor_type&;
	auto enqueue(transaction tr) -> tr_result;
	auto enqueue(launch_async_t, transaction tr) -> void;
	auto stop_queue(bool wait_exit) -> void;

	// server actor management
	auto toggle(bool on) -> error;

	auto start_server() -> void;

	auto start_client(const std::string& host) -> error;

	auto publish_link(tree::link L) -> error;
	auto unpublish_link(tree::lid_type lid) -> error;

private:
	// kernel's actor system
	// delayed actor system initialization
	std::optional<caf::actor_system> actor_sys_;

	// actor_system getter switched at runtime on kernel init/shutdown
	using as_getter_f = caf::actor_system& (radio_subsyst::*)();
	as_getter_f get_actor_sys_;
	auto normal_as_getter() -> caf::actor_system&;
	auto always_throw_as_getter() -> caf::actor_system&;

	// storage for actor addresses
	using citizens_registry_t = std::unordered_set<caf::actor_addr>;
	citizens_registry_t citizens_;
	caf::detail::shared_spinlock guard_;

	// queue
	kqueue_actor_type queue_;

	caf::actor radio_;

	auto reset_timeouts(timespan typical, timespan slow) -> void;
	auto spawn_queue() -> void;
};

NAMESPACE_END(blue_sky::kernel::detail)
