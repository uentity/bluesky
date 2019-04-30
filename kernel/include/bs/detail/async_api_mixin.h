/// @file
/// @author uentity
/// @date 16.08.2018
/// @brief Pattern of async API call using CAF actors
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../kernel/config.h"
#include <caf/scoped_actor.hpp>
#include <caf/send.hpp>

namespace blue_sky::detail {

/// Mixin that allows bi-directional communication with controlled actor
template<class Derived>
class async_api_mixin {
private:
	template<typename A1 = int, typename... As>
	static constexpr bool a1_is_priority = std::is_same_v<caf::message_priority, std::decay_t<A1>>;

public:
	using message_priority = caf::message_priority;
	explicit async_api_mixin() : sender_(kernel::config::actor_system(), true) {}

	// link sender with target actor, so they die together
	auto init_sender() const -> void {
		sender_->link_to(derived().actor());
	}

	// message priority can be passed as template arg
	template<
		message_priority P = message_priority::normal, typename... Args
	>
	auto send(Args&&... args) const -> std::enable_if_t<!a1_is_priority<Args...>> {
		caf::anon_send<P>(derived().actor(), std::forward<Args>(args)...);
	}

	// message priority passed as first argument
	template<typename... Args>
	void send(message_priority prio, Args&&... args) const {
		if(prio == message_priority::high)
			caf::anon_send<message_priority::high>(derived().actor(), std::forward<Args>(args)...);
		else
			caf::anon_send<message_priority::normal>(derived().actor(), std::forward<Args>(args)...);
	}

	auto sender() const -> const caf::scoped_actor& {
		return sender_;
	}

private:
	caf::scoped_actor sender_;
	inline auto derived() const -> const Derived& { return static_cast<const Derived&>(*this); }
};

/// Lightweight async mixin:
// 1. don't carry blocking actor, sends messages using `caf::anon_send`
// 2. cannot receive any result from destination actor, i.e. it's one way connection
// 3. sends `exit` (kill) message on destruction to terminate corresponding actor
template<class ActorT>
struct anon_async_api_mixin {
private:
	template<typename A1 = int, typename... As>
	static constexpr bool a1_is_priority = std::is_same_v<caf::message_priority, std::decay_t<A1>>;

public:
	ActorT actor;

	template<
		caf::spawn_options Os = caf::no_spawn_options, typename F,
		typename = std::enable_if_t<!std::is_base_of_v<anon_async_api_mixin, std::decay_t<F>>>
	>
	explicit anon_async_api_mixin(F&& async_behavior) {
		spawn<Os>(std::forward<F>(async_behavior));
	}
	// empty ctor - don't spawn actor
	anon_async_api_mixin() {}
	// terminate controlled actor when this (master) instance is dying
	~anon_async_api_mixin() {
		caf::anon_send_exit(actor, caf::exit_reason::kill);
	}

	anon_async_api_mixin(const anon_async_api_mixin&) = default;
	anon_async_api_mixin(anon_async_api_mixin&&) = default;
	anon_async_api_mixin& operator=(anon_async_api_mixin&&) = default;

	template<caf::spawn_options Os = caf::no_spawn_options, typename F>
	auto& spawn(F&& async_behavior) {
		actor = kernel::config::actor_system().spawn<Os>(std::forward<F>(async_behavior));
		return actor;
	}

	// message priority can be passed as template arg
	using message_priority = caf::message_priority;
	template<
		message_priority P = message_priority::normal, typename... Args
	>
	auto send(Args&&... args) const -> std::enable_if_t<!a1_is_priority<Args...>> {
		caf::anon_send<P>(actor, std::forward<Args>(args)...);
	}

	// message priority passed as first argument
	template<typename... Args>
	void send(message_priority prio, Args&&... args) const {
		if(prio == message_priority::high)
			caf::anon_send<message_priority::high>(actor, std::forward<Args>(args)...);
		else
			caf::anon_send<message_priority::normal>(actor, std::forward<Args>(args)...);
	}
};
	
} /* namespace blue_sky::detail */

