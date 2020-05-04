/// @file
/// @author uentity
/// @date 30.03.2020
/// @brief BS tree link events handling
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "link_actor.h"
#include "ev_listener_actor.h"
#include <bs/kernel/radio.h>
#include <bs/log.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

NAMESPACE_BEGIN(blue_sky::tree)

auto link::subscribe(handle_event_cb f, Event listen_to) const -> std::uint64_t {
	using namespace kernel::radio;
	using namespace allow_enumops;
	using baby_t = ev_listener_actor<link>;

	// produce event bhavior that calls passed callback with proper params
	auto make_ev_character = [weak_src = weak_ptr{*this}, src_id = id(), listen_to](baby_t* self) {
		auto res = caf::message_handler{};
		if(enumval(listen_to & Event::LinkRenamed))
			res = res.or_else(
				[=](a_ack, a_lnk_rename, std::string new_name, std::string old_name) {
					if(auto src = weak_src.lock())
						self->f(std::move(src), Event::LinkRenamed, {
							{"new_name", std::move(new_name)},
							{"prev_name", std::move(old_name)}
						});
				}
			);

		if(enumval(listen_to & Event::LinkStatusChanged))
			res = res.or_else(
				[=](a_ack, a_lnk_status, Req request, ReqStatus new_v, ReqStatus prev_v) {
					if(auto src = weak_src.lock())
						self->f(std::move(src), Event::LinkStatusChanged, {
							{"request", prop::integer(new_v)},
							{"new_status", prop::integer(new_v)},
							{"prev_status", prop::integer(prev_v)}
						});
				}
			);

		if(enumval(listen_to & Event::LinkDeleted))
			res = res.or_else(
				[=](a_bye) {
					// when overriding `a_bye` we must quit explicitly
					// [NOTE] terminate self behavior, but current handler will execute till end
					self->quit();
					// do callback job
					if(auto src = weak_src.lock())
						self->f( std::move(src), Event::LinkDeleted, {{ "lid", to_string(src_id) }} );
				}
			);

		return res;
	};

	// make baby event handler actor
	auto baby = system().spawn_in_group<baby_t>(
		pimpl_->home, pimpl_->home, std::move(f), std::move(make_ev_character)
	);
	// return baby ID
	return baby.id();
}

auto link::unsubscribe(std::uint64_t event_cb_id) -> void {
	kernel::radio::bye_actor(event_cb_id);
}

NAMESPACE_END(blue_sky::tree)
