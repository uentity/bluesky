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
// interface of kernel's home group
using khome_actor_type = caf::typed_actor<
	caf::reacts_to<a_bye>
>;

// kernel's queue interface
using kqueue_actor_type = caf::typed_actor<
	caf::replies_to<simple_transaction>::with<error::box>
>::extend_with<khome_actor_type>;

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

	// post transaction into Python's queue
	auto enqueue(simple_transaction tr) -> error;
	auto enqueue(launch_async_t, simple_transaction tr) -> void;
	auto queue_actor() -> kqueue_actor_type&;

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
	struct queue_handle {
		kqueue_actor_type actor;
		caf::scoped_actor factor;

		queue_handle();
	};
	std::optional<queue_handle> queue_;
	// lazy start & access queue actor
	auto queue() -> queue_handle&;

	caf::actor radio_;

	auto reset_timeouts(timespan typical, timespan slow) -> void;
};

NAMESPACE_END(blue_sky::kernel::detail)
