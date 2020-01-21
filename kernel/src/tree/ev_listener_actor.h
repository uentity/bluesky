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
#include <bs/propdict.h>
#include <bs/tree/common.h>
#include <bs/kernel/radio.h>

#include <caf/group.hpp>
#include <caf/actor_config.hpp>
#include <caf/event_based_actor.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

template<typename Source>
struct ev_listener_actor : caf::event_based_actor {
	using super = caf::event_based_actor;
	using behavior_type = super::behavior_type;

	// event processor callback
	using callback_t = std::function< void(Source, Event, prop::propdict) >;

	static auto make_safe_callback(callback_t&& f) {
		return [f = std::move(f)](Source L, Event ev, prop::propdict props) -> void {
			error::eval_safe([&]{ f(std::move(L), ev, std::move(props)); });
		};
	}

	// safe callback wrapped into error::eval_safe()
	using safe_callback_t = decltype(make_safe_callback(std::declval<callback_t>()));
	safe_callback_t f;

	// listen to messages from this group
	caf::group grp;

	ev_listener_actor(
		caf::actor_config& cfg, caf::group tgt_grp, callback_t cb,
		std::function< caf::message_handler(ev_listener_actor*) > make_event_behavior
	)
		: super(cfg), grp(std::move(tgt_grp)), f(make_safe_callback(std::move(cb)))
	{
		// silently drop all other messages not in my character
		set_default_handler(caf::drop);
		// self-register
		kernel::radio::system().registry().put(id(), this);

		// generate & remember self behavior
		character = make_event_behavior(this).or_else(caf::message_handler{
			[=](a_bye) { disconnect(); }
		});
	}

	auto disconnect() -> void {
		leave(grp);
		kernel::radio::system().registry().erase(id());
	}

	auto make_behavior() -> behavior_type override {
		return std::move(character);
	}

private:
	// temp storage for event handlers between ctor and make_behavior()
	caf::behavior character;
};

NAMESPACE_END(blue_sky::tree)
