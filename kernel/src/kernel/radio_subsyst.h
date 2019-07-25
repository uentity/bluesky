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
#include <caf/actor_system.hpp>

#include <mutex>

NAMESPACE_BEGIN(blue_sky::kernel::detail)

struct BS_HIDDEN_API radio_subsyst {
	// store links that will be visible to the world
	std::set< tree::sp_link, std::less<> > publinks;

	radio_subsyst();

	auto init() -> error;
	auto shutdown() -> void;

	auto system() -> caf::actor_system&;

	auto toggle(bool on) -> error;

	auto start_server() -> void;

	auto start_client(const std::string& host) -> error;

	auto publish_link(tree::sp_link L) -> error;
	auto unpublish_link(tree::link::id_type lid) -> error;

private:
	// kernel's actor system
	// delayed actor system initialization
	std::optional<caf::actor_system> actor_sys_;

	caf::actor radio_;
};

NAMESPACE_END(blue_sky::kernel::detail)
