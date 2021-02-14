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

	// immediate save object
	[=](a_save, sp_obj obj, const std::string& fmt_name, std::string fname) -> error::box {
		// obtain formatter
		auto F = get_formatter(obj->type_id(), fmt_name);
		if(!F) return error{obj->type_id(), tree::Error::MissingFormatter};

		// run save job
		// not using transaction as saving must not trigger DataModified event
		return error::eval_safe([&] {
			//caf::aout(this) << "Saving " << fname << std::endl;
			return F->save(*obj, fname);
		});
	},

	// immediate load
	[=](a_load, sp_obj obj, const std::string& fmt_name, std::string fname) -> error::box {
		// obtain formatter
		auto F = get_formatter(obj->type_id(), fmt_name);
		if(!F) return error{obj->type_id(), tree::Error::MissingFormatter};

		// apply load job
		// not using transaction as loading must not trigger DataModified event
		return error::eval_safe([&] {
			//caf::aout(this) << "Loading " << fname << std::endl;
			return F->load(*obj, fname);
		});
	},

	// lazy load
	[=](a_load, const sp_obj&) -> error::box { return success(); },

	// setup lazy load
	[=](a_lazy, a_load, const std::string& fmt_name, const std::string& fname) {
		auto orig_me = current_behavior();
		become(caf::message_handler{
			// 1. patch lazy load request to actually trigger reading from file
			[=](a_load, sp_obj obj) mutable -> error::box {
				// trigger only once
				become(orig_me);
				// apply load job inplace
				return actorf<error::box>(
					orig_me, a_load(), std::move(obj), std::move(fmt_name), std::move(fname)
				);
			},

			// 2. patch 'normal load' to drop lazy load behavior
			[=](a_load, sp_obj obj, std::string cur_fmt, std::string cur_fname)
			mutable -> error::box {
				become(orig_me);
				return actorf<error::box>(
					orig_me, a_load(), std::move(obj), std::move(cur_fmt), std::move(cur_fname)
				);
			},

			// 3. patch 'a_save' request to be noop - until object will be actually read
			// this also means that if object is not yet loaded, then saving to same file is noop
			[=](a_save, const sp_obj& obj, std::string cur_fmt, std::string cur_fname)
			mutable -> error::box {
				// noop if saving to same file with same format
				// otherwise invoke lazy load (read object) & then save it
				if(cur_fmt == fmt_name && cur_fname == fname)
					return success();
				else {
					// [NOTE] need `current_behavior()` because lazy load is noop in `orig_me`
					if(auto er = actorf<error::box>(current_behavior(), a_load(), obj); er.ec)
						return er;
					// apply save job inplace
					// [NOTE] use `orig_me` here, because it have unpatched (normal) save
					return actorf<error::box>(
						orig_me, a_save(), std::move(obj), std::move(cur_fmt), std::move(cur_fname)
					);
				}
			}
		}.or_else(orig_me));
		return true;
	},

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
