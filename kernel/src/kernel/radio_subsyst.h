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

#include <caf/actor.hpp>
#include <caf/typed_actor.hpp>
#include <caf/actor_system.hpp>
#include <caf/group.hpp>

#include <optional>
#include <set>

#define KRADIO ::blue_sky::singleton<::blue_sky::kernel::detail::radio_subsyst>::Instance()

NAMESPACE_BEGIN(blue_sky::kernel::detail)
// interface of kernel's home group
using khome_actor_type = caf::typed_actor<
	caf::reacts_to<a_bye>
>;

struct BS_HIDDEN_API radio_subsyst {
	// get kernel's home group
	auto khome() -> const caf::group&;

	// store links that will be visible to the world
	std::set< tree::link, std::less<> > publinks;

	radio_subsyst();

	auto init() -> error;
	auto shutdown() -> void;

	inline auto system() -> caf::actor_system& {
		return (this->*get_actor_sys_)();
	}

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

	// kernel's home group
	caf::group khome_;

	caf::actor radio_;
};

NAMESPACE_END(blue_sky::kernel::detail)
