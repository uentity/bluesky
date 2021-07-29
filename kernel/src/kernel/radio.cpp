/// @file
/// @author uentity
/// @date 24.07.2019
/// @brief Kernel radio subsystem impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel/radio.h>
#include "radio_subsyst.h"
#include "kimpl.h"

#define KRADIO ::blue_sky::singleton<::blue_sky::kernel::detail::radio_subsyst>::Instance()

NAMESPACE_BEGIN(blue_sky::kernel::radio)

auto system() -> caf::actor_system& {
	return KRADIO.system();
}

auto bye_actor(std::uint64_t actor_id) -> bool {
	const auto ev_actor = system().registry().get(actor_id);
	// [NOTE] need to do `actor_cast` to resolve `send()` resolution ambiguity
	const auto A = caf::actor_cast<caf::actor>(ev_actor);
	caf::anon_send<caf::message_priority::high>(A, a_bye());
	return static_cast<bool>(A);
}

auto toggle(bool on) -> void {
	KRADIO.toggle(on);
}

auto start_server() -> void {
	KRADIO.start_server();
}

auto start_client(const std::string& host) -> blue_sky::error {
	return KRADIO.start_client(host);
}

auto publish_link(tree::link L) -> error {
	return KRADIO.publish_link(std::move(L));
}

auto unpublish_link(tree::link L) -> error {
	return KRADIO.unpublish_link(L.id());
}

NAMESPACE_END(blue_sky::kernel::radio)
