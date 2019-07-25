/// @file
/// @author uentity
/// @date 24.07.2019
/// @brief BS kernel radio subsystem impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/atoms.h>
#include <bs/log.h>
#include <bs/kernel/config.h>
#include <bs/kernel/misc.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>
#include "radio_subsyst.h"

#include <caf/actor_system_config.hpp>
#include <caf/event_based_actor.hpp>
#include <caf/typed_event_based_actor.hpp>
#include <caf/io/middleman.hpp>

#include <iostream>

#define BSCONFIG ::blue_sky::kernel::config::config()
#define KRADIO ::blue_sky::singleton<::blue_sky::kernel::detail::radio_subsyst>::Instance()

inline constexpr std::uint16_t def_port = 9339;
inline constexpr std::uint16_t def_groups_port = 9340;
inline constexpr blue_sky::timespan def_timeout = std::chrono::seconds(30);

NAMESPACE_BEGIN(blue_sky::kernel::detail)
using namespace kernel::config;

/*-----------------------------------------------------------------------------
 *  BS main network actor
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()

using radio_station_handle = caf::typed_actor<
	//caf::replies_to< a_hi >::with< std::vector<tree::link::id_type> >
	caf::replies_to< a_hi >::with< std::vector<tree::sp_link> >
>;

auto radio_station(radio_station_handle::pointer self)
-> radio_station_handle::behavior_type { return {
	[](a_hi) {
		std::vector<tree::sp_link> publids;
		for(const auto& L : KRADIO.publinks)
			publids.push_back(L);
		//std::vector<tree::link::id_type> publids;
		//for(const auto& L : KRADIO.publinks)
		//	publids.push_back(L->id());
		return publids;
	}
}; }

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  radio subsystem impl
 *-----------------------------------------------------------------------------*/
radio_subsyst::radio_subsyst() {
	actor_config()
		.add_actor_type("radio_station", radio_station)
		.add_message_type<tree::link::id_type>("link_id_type")
	;

	// [NOTE] `actor_system` starts worker and other service threads in constructor.
	// BS kernel singleton is constructed during initialization of kernel shared library.
	// And on Windows it is PROHIBITED to start threads in `DllMain()`, because that cause a deadlock.
	// Solution: delay construction of actor_system until first usage, don't use CAf in kernel ctor.

	// [NOTE] radio subsystem is explicitly created by `kernel::misc::init()`,
	// so it's safe to init `actor_system` here
	if(auto er = init(); er)
		throw er;
}

auto radio_subsyst::init() -> error {
	if(!actor_sys_) {
		// load network module
		actor_config().load<caf::io::middleman>();
		// start actor system
		if(actor_sys_.emplace(actor_config()); !actor_sys_)
			return error{ "Can't create CAF actor_system!" };
	}
	return perfect;
}

auto radio_subsyst::shutdown() -> void {
	if(actor_sys_) actor_sys_.reset();
}

auto radio_subsyst::system() -> caf::actor_system& {
	// ensure actor system is initialized
	static auto& actor_sys = [=]() -> caf::actor_system& {
		if(auto er = init(); er)
			throw er;
		return *actor_sys_;
	}();
	return actor_sys;
}

auto radio_subsyst::toggle(bool on) -> error {
	auto& mm = actor_sys_->middleman();
	auto port = get_or(BSCONFIG, "port", def_port);
	auto g_port = get_or(BSCONFIG, "groups-port", def_groups_port);

	if(on) {
		// open port for remote spawn
		auto res = mm.open(port, nullptr, true);
		if(!res) return { actor_sys_->render(res.error()) };
		// publish local groups
		res = mm.publish_local_groups(g_port, nullptr, true);
		if(!res) return { actor_sys_->render(res.error()) };
	}
	else {
		mm.close(port);
		mm.close(g_port);
	}

	return perfect;
}

auto radio_subsyst::start_server() -> void {
	if(toggle(true)) return;
	actor_config().add_message_type<tree::sp_link>("link");

	std::cout << "*** started server on port " << get_or(BSCONFIG, "port", def_port) << std::endl
		<< "type 'quit' to shutdown the server" << std::endl;
	std::string line;
	while (getline(std::cin, line)) {
		if (line == "quit")
			return;
		else
			std::cerr << "illegal command" << std::endl;
	}
	// stop server - just close ports
	toggle(false);
}

auto radio_subsyst::start_client(const std::string& host) -> error {
	auto netnode = actor_sys_->middleman().connect(host, get_or(BSCONFIG, "port", def_port));
	if(!netnode) return { actor_sys_->render(netnode.error()) };
	else {
		bsout() << "Successfully connected to '{}:{}'" <<
			host << get_or(BSCONFIG, "port", def_port) << bs_end;
	}

	actor_config().add_message_type<tree::sp_link>("link");
	auto station = actor_sys_->middleman().remote_spawn<radio_station_handle>(
		*netnode, "radio_station", caf::make_message(), get_or(BSCONFIG, "timeout", def_timeout)
	);
	if(!station) return { actor_sys_->render(station.error()) };

	// get published links
	auto station_f = caf::make_function_view(*station);
	auto res = station_f(a_hi());
	if(!res) return { "Failed to retrive public links from server" };

	std::cout << "Server returned public links: [";
	auto first = true;
	for(const auto& lid : *res) {
		if(first) first = false;
		else std::cout << ' ';
		std::cout << to_string(lid->id());
	}
	std::cout << ']' << std::endl;

	// need to do this otherwise server hangs
	caf::anon_send_exit(*station, caf::exit_reason::kill);
	return perfect;
}

auto radio_subsyst::publish_link(tree::sp_link L) -> error {
	publinks.insert(std::move(L));
	return perfect;
}

auto radio_subsyst::unpublish_link(tree::link::id_type lid) -> error {
	for(auto L = publinks.begin(), end = publinks.end(); L != end; ++L)
		if((*L)->id() == lid) publinks.erase(L);
	return perfect;
}

NAMESPACE_END(blue_sky::kernel::detail)
