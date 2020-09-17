/// @file
/// @author uentity
/// @date 10.03.2020
/// @brief Implementation of objbase actor
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/objbase.h>
#include <bs/defaults.h>
#include <bs/actor_common.h>
#include <bs/uuid.h>
#include <bs/tree/common.h>
#include <bs/kernel/config.h>
#include <bs/kernel/radio.h>
#include <bs/detail/scope_guard.h>

#include <bs/serialize/cafbind.h>
#include <bs/serialize/object_formatter.h>
#include <bs/serialize/propdict.h>

#include "objbase_actor.h"

#include <caf/actor_ostream.hpp>

NAMESPACE_BEGIN(blue_sky)
using namespace kernel::radio;
using namespace std::chrono_literals;

/*-----------------------------------------------------------------------------
 *  objbase_actor
 *-----------------------------------------------------------------------------*/
objbase_actor::objbase_actor(caf::actor_config& cfg, caf::group home) :
	super(cfg), home_(std::move(home))
{
	// exit after kernel
	KRADIO.register_citizen(this);
}

auto objbase_actor::make_typed_behavior() -> typed_behavior {
return typed_behavior {
	[=](a_bye) { if(current_sender() != this) quit(); },

	// get home group
	[=](a_home) { return home_; },

	// rebind to new home
	[=](a_home, const std::string& new_hid) {
		// leave old home
		leave(home_);
		send(home_, a_bye());
		// enter new one
		home_ = system().groups().get_local(new_hid);
		join(home_);
	},

	// execute transaction
	[=](a_apply, const transaction& m) -> tr_result::box {
		auto tres = pack(tr_eval(m));
		send(home_, a_ack(), a_data(), tres);
		return tres;
	},

	// skip acks - sent by myself
	[=](a_ack, a_data, const tr_result::box&) {},

	[=](a_delay_load, std::string fmt_name, std::string fname) {
		auto cur_me = current_behavior();
		become(caf::message_handler{
			[=, fmt_name = std::move(fmt_name), fname = std::move(fname)](a_delay_load, sp_obj obj)
			mutable -> error::box {
				// trigger only once
				become(cur_me);
				// obtain formatter
				auto F = get_formatter(obj->type_id(), fmt_name);
				if(!F) return error{obj->type_id(), tree::Error::MissingFormatter};

				// apply load job inplace
				auto job = caf::make_message(a_apply(), transaction{
					[=, obj = std::move(obj), fname = std::move(fname)] {
						//caf::aout(this) << "Loading " << fname << std::endl;
						return F->load(*obj, fname);
					}
				});
				auto res = error::box(quiet_fail);
				cur_me(job)->extract(
					[&](error::box er) { res = std::move(er); }
				);
				return res;
			}
		}.or_else(cur_me));
		return true;
	},

	[=](a_delay_load, const sp_obj&) -> error::box { return success(); }

}; }

auto objbase_actor::make_behavior() -> behavior_type {
	return make_typed_behavior().unbox();
}

auto objbase_actor::on_exit() -> void {
	// say bye-bye to self group
	send(home_, a_bye());

	KRADIO.release_citizen(this);
}

/*-----------------------------------------------------------------------------
 *  objbase
 *-----------------------------------------------------------------------------*/
auto objbase::start_engine() -> bool {
	if(!actor_) {
		if(!home_) reset_home({}, true);
		actor_ = system().spawn_in_group<objbase_actor>(home_, home_);
		return true;
	}
	return false;
}

objbase::~objbase() {
	// explicitly stop engine
	caf::anon_send_exit(actor_, caf::exit_reason::user_shutdown);
}

auto objbase::home() const -> const caf::group& { return home_; }

auto objbase::home_id() const -> std::string_view {
	return home_ ? home_.get()->identifier() : std::string_view{};
}

auto objbase::reset_home(std::string new_hid, bool silent) -> void {
	if(new_hid.empty()) new_hid = to_string(gen_uuid());
	// send home rebind message to old home (not directly to actor to also inform hard_links)
	if(!silent)
		checked_send<objbase_actor::home_actor_type, high_prio>(home_, a_home(), new_hid);
	home_ = system().groups().get_local(new_hid);
}

auto objbase::apply(transaction m) const -> tr_result {
	return actorf<tr_result>(
		actor(), kernel::radio::timeout(true), a_apply(), std::move(m)
	);
}

auto objbase::apply(obj_transaction tr) const -> tr_result {
	return actorf<tr_result>(
		actor(), kernel::radio::timeout(true), a_apply(), make_transaction(std::move(tr))
	);
}

auto objbase::apply(launch_async_t, transaction m) const -> void {
	caf::anon_send(actor(), a_apply(), std::move(m));
}

auto objbase::apply(launch_async_t, obj_transaction tr) const -> void {
	caf::anon_send(actor(), a_apply(), make_transaction(std::move(tr)));
}

auto objbase::touch(tr_result tres) const -> void {
	caf::anon_send(actor(), a_apply(), transaction{
		[tres = std::move(tres)]() mutable { return std::move(tres); }
	});
}

NAMESPACE_END(blue_sky)
