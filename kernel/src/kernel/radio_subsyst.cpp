/// @file
/// @author uentity
/// @date 24.07.2019
/// @brief BS kernel radio subsystem impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "radio_subsyst.h"
#include "../tree/nil_engine.h"

#include <bs/actor_common.h>
#include <bs/log.h>
#include <bs/kernel/config.h>
#include <bs/kernel/misc.h>

#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

#include <caf/actor_system_config.hpp>
#include <caf/typed_event_based_actor.hpp>
#include <caf/io/middleman.hpp>

#include <iostream>

#define BSCONFIG ::blue_sky::kernel::config::config()

inline constexpr std::uint16_t def_port = 9339;
inline constexpr std::uint16_t def_groups_port = 9340;

NAMESPACE_BEGIN(blue_sky::kernel::detail)
using namespace kernel::config;
using namespace std::chrono_literals;

/*-----------------------------------------------------------------------------
 *  BS main network actor
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()

using radio_station_handle = caf::typed_actor<
	//caf::replies_to< a_hi >::with< std::vector<tree::lid_type> >
	caf::replies_to< a_hi >::with< std::vector<tree::link> >
>;

auto radio_station(radio_station_handle::pointer self)
-> radio_station_handle::behavior_type { return {
	[](a_hi) {
		std::vector<tree::link> publids;
		for(const auto& L : KRADIO.publinks)
			publids.push_back(L);
		//std::vector<tree::lid_type> publids;
		//for(const auto& L : KRADIO.publinks)
		//	publids.push_back(L->id());
		return publids;
	}
}; }

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  radio subsystem impl
 *-----------------------------------------------------------------------------*/
radio_subsyst::radio_subsyst() : get_actor_sys_(&radio_subsyst::always_throw_as_getter)
{
	actor_config()
		.add_actor_type("radio_station", radio_station)
		.add_message_type<tree::lid_type>("link_id_type")
	;

	if(auto er = init(); er)
		throw er;
}

auto radio_subsyst::init() -> error {
	if(!actor_sys_) {
		// kernel must be configured (middleman module loaded)
		if(!kernel::config::is_configured())
			kernel::config::configure();

		// start actor system
		if(actor_sys_.emplace(actor_config()); !actor_sys_)
			return error{ "Can't create CAF actor_system!" };
		get_actor_sys_ = &radio_subsyst::normal_as_getter;

		spawn_queue();
	}
	return perfect;
}

auto radio_subsyst::shutdown() -> void {
	if(!actor_sys_) return;
	std::cout << "~~~ radio shutdown start" << std::endl;

	// force all pending requests to exit very quickly
	reset_timeouts(1us, 1us);
	kick_citizens();

	stop_queue(false);

	// explicitly kill nill link
	tree::nil_node::stop();
	tree::nil_link::stop();

	// explicit wait until all actors done if asked for
	// because during termination some actor may need to access live actor_system
	if(get_or(config::config(), "radio.await_actors_before_shutdown", true)) {
		std::cout << "~~~ Waiting for " << actor_sys_->registry().running() << " actors" << std::endl;
		actor_sys_->await_all_actors_done();
		std::cout << "~~~ Waiting actors finished" << std::endl;
	}

	// destroy actor_system
	actor_sys_->await_actors_before_shutdown(false);
	get_actor_sys_ = &radio_subsyst::always_throw_as_getter;
	actor_sys_.reset();
	std::cout << "~~~ radio shutdown finished" << std::endl;
}

auto radio_subsyst::normal_as_getter() -> caf::actor_system& {
	return *actor_sys_;
}
auto radio_subsyst::always_throw_as_getter() -> caf::actor_system& {
	throw error{"Kernel's radio subsystem is down"};
}

auto radio_subsyst::register_citizen(caf::actor_addr citizen) -> void {
	auto solo = std::lock_guard{guard_};
	citizens_.insert(std::move(citizen));
}

auto radio_subsyst::release_citizen(const caf::actor_addr& citizen) -> void {
	auto solo = std::lock_guard{guard_};
	citizens_.erase(citizen);
}

auto radio_subsyst::kick_citizens() -> void {
	const auto kick_out = [&] {
		auto waiter = caf::scoped_actor{system(), false};
		std::vector<caf::actor> alive;
		{
			auto solo = std::lock_guard{guard_};
			if(citizens_.empty()) return false;

			alive.reserve(citizens_.size());
			for(const auto& caddr : citizens_) {
				if(auto A = caf::actor_cast<caf::actor>(caddr)) {
					waiter->send_exit(A, caf::exit_reason::kill);
					alive.push_back(std::move(A));
				}
			}
			citizens_.clear();
		}
		// wait until citizen gone
		std::cout << "~~~ kick out " << alive.size() << " actors" << std::endl;
		waiter->wait_for(std::move(alive));
		std::cout << "~~~ kick out finished" << std::endl;
		return true;
	};
	while(kick_out()) {};
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

///////////////////////////////////////////////////////////////////////////////
//  server management
//
auto radio_subsyst::start_server() -> void {
	if(toggle(true)) return;
	actor_config().add_message_type<tree::link>("link");

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

	actor_config().add_message_type<tree::link>("link");
	auto station = actor_sys_->middleman().remote_spawn<radio_station_handle>(
		*netnode, "radio_station", caf::make_message(), kernel::radio::timeout(true)
	);
	if(!station) return { actor_sys_->render(station.error()) };

	// get published links
	auto station_f = caf::make_function_view(*station);
	auto res = station_f(a_hi());
	if(!res) return { "Failed to retrive public links from server" };

	std::cout << "Server returned public links: [";
	auto first = true;
	for(const auto& L : *res) {
		if(first) first = false;
		else std::cout << ' ';
		std::cout << to_string(L.id());
	}
	std::cout << ']' << std::endl;

	// need to do this otherwise server hangs
	caf::anon_send_exit(*station, caf::exit_reason::kill);
	return perfect;
}

auto radio_subsyst::publish_link(tree::link L) -> error {
	publinks.insert(std::move(L));
	return perfect;
}

auto radio_subsyst::unpublish_link(tree::lid_type lid) -> error {
	for(auto L = publinks.begin(), end = publinks.end(); L != end; ++L)
		if(L->id() == lid) publinks.erase(L);
	return perfect;
}

NAMESPACE_END(blue_sky::kernel::detail)
