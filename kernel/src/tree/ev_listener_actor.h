/// @file
/// @author uentity
/// @date 21.01.2020
/// @brief Boilerplate for making events processing acctors
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/atoms.h>
#include <bs/error.h>

#include "private_common.h"
#include "../kernel/radio_subsyst.h"

#include <caf/actor_config.hpp>
#include <caf/event_based_actor.hpp>

#include <optional>

NAMESPACE_BEGIN(blue_sky::tree)

template<typename Source>
struct ev_listener_actor : caf::event_based_actor {
	using super = caf::event_based_actor;
	using behavior_type = super::behavior_type;

	using actor_type = ev_listener_actor_type;

	// event processor callback
	using callback_t = typename Source::event_handler;

	static auto make_safe_callback(callback_t&& f) {
		return callback_t{[f = std::move(f)](auto&&... xs) -> void {
			error::eval_safe([&]{ f(std::forward<decltype(xs)>(xs)...); });
		}};
	}

	// safe callback wrapped into error::eval_safe()
	using safe_callback_t = decltype(make_safe_callback(std::declval<callback_t>()));
	std::optional<safe_callback_t> f_;
	const safe_callback_t& f;

	// address of events source actor (engine)
	const caf::actor_addr origin;

	ev_listener_actor(
		caf::actor_config& cfg, caf::actor_addr ev_src, callback_t cb,
		std::function< caf::message_handler(ev_listener_actor*) > make_event_behavior
	)
		: super(cfg), f_(make_safe_callback(std::move(cb))), f(*f_), origin(std::move(ev_src))
	{
		// silently drop all other messages not in my character
		set_default_handler(noop_r<caf::message>());

		// [NOTE] lifetime of actor is kept by group it joins on 'a_hi'
		// exit after kernel
		KRADIO.register_citizen(this);
		// next is required to convert from std::uint64_t actor id -> actor handle
		KRADIO.system().registry().put(id(), this);

		// generate & remember self behavior
		character = make_event_behavior(this).or_else(actor_type::behavior_type{
			// ping-pong check that actor started
			[=](a_hi, const caf::group& src_home) {
				join(src_home);
				return id();
			},
			// exit on a_bye
			[=](a_bye) { quit(); }
		}.unbox());
	}

	auto name() const -> const char* override { return "ev_listener_actor"; }

	auto make_behavior() -> behavior_type override {
		return std::move(character);
	}

	auto on_exit() -> void override {
		// destroy callback as early as possible
		f_.reset();
		KRADIO.release_citizen(this);
	}

private:
	// temp storage for event handlers between ctor and make_behavior()
	caf::behavior character;
};

NAMESPACE_END(blue_sky::tree)
