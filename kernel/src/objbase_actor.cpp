/// @file
/// @author uentity
/// @date 10.03.2020
/// @brief Implementation of objbase actor
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "objbase_actor.h"

#include <bs/actor_common.h>
#include <bs/defaults.h>
#include <bs/objbase.h>
#include <bs/uuid.h>
#include <bs/type_caf_id.h>

#include <bs/kernel/config.h>
#include <bs/kernel/radio.h>

#include <bs/serialize/object_formatter.h>

#include <caf/actor_ostream.hpp>

NAMESPACE_BEGIN(blue_sky)
using namespace kernel::radio;
using namespace std::chrono_literals;

/*-----------------------------------------------------------------------------
 *  objbase_actor
 *-----------------------------------------------------------------------------*/
objbase_actor::objbase_actor(caf::actor_config& cfg, sp_obj mama) :
	super(cfg), home_(mama->home()), mama_(mama)
{
	// exit after kernel
	KRADIO.register_citizen(this);
}

auto objbase_actor::name() const -> const char* { return "objbase actor"; }

auto objbase_actor::make_typed_behavior() -> typed_behavior {
return typed_behavior {
	// ignore `a_bye` signal - comes from self
	[=](a_bye) {},

	// get home group
	[=](a_home) { return home_; },

	// execute transaction
	[=](a_apply, const obj_transaction& otr) -> caf::result<tr_result::box> {
		// if transaction is async, go through additional request,
		// because we have to deliver notification
		if(carry_async_transaction(otr)) {
			auto tres = make_response_promise<tr_result::box>();
			request(caf::actor_cast<actor_type>(this), caf::infinite, a_ack{}, a_apply{}, otr)
			.then(
				[=](tr_result::box res) mutable {
					send(home_, a_ack(), a_data(), res);
					tres.deliver(std::move(res));
				},
				[=](const caf::error& er) mutable {
					auto res = pack(tr_result(forward_caf_error(er)));
					send(home_, a_ack(), a_data(), res);
					tres.deliver(std::move(res));
				}
			);
			return tres;
		}
		// ohtherwise eval directly
		else {
			auto tres = tr_eval(this, otr, [&] { return mama_.lock(); });
			// non-async transaction must fill 'message' slot of caf::result
			visit(
				[&](auto& mres) {
					if constexpr(std::is_same_v<meta::remove_cvref_t<decltype(mres)>, caf::message>) {
						auto notify = caf::behavior{[&](const tr_result::box& rb) {
							send(home_, a_ack(), a_data(), rb);
						}};
						notify(mres);
					}
				},
				tres.get_data()
			);
			return tres;
		}
	},
	// extra handler to exec async transaction
	[=](a_ack, a_apply, const obj_transaction& tr) -> caf::result<tr_result::box> {
		return tr_eval(this, tr, [&] { return mama_.lock(); });
	},

	// skip acks - sent by myself
	[=](a_ack, a_data, const tr_result::box&) {},

	// immediate save object
	[=](a_save, const std::string& fmt_name, std::string fname) -> error::box {
		auto obj = mama_.lock();
		if(!obj) return error::quiet(tree::Error::EmptyData);
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
	[=](a_load, const std::string& fmt_name, std::string fname) -> error::box {
		auto obj = mama_.lock();
		if(!obj) return error::quiet(tree::Error::EmptyData);
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
	[=](a_load) -> error::box { return success(); },

	[=](a_lazy, a_load, a_data_node) { return false; },

	// setup lazy load
	[=](a_lazy, a_load, const std::string& fmt_name, const std::string& fname, bool with_node) {
		auto orig_me = current_behavior();
		become(caf::message_handler{
			// return remembered flag whether to read node from file
			[=](a_lazy, a_load, a_data_node) { return with_node; },

			// 1. patch lazy load request to actually trigger reading from file
			[=](a_load) mutable -> error::box {
				// trigger only once
				become(orig_me);
				// invoke load job inplace on `orig_me` behavior
				return actorf<error::box>(
					orig_me, a_load(), std::move(fmt_name), std::move(fname)
				);
			},

			// 2. patch 'normal load' to drop lazy load behavior
			[=](a_load, std::string cur_fmt, std::string cur_fname)
			mutable -> error::box {
				become(orig_me);
				return actorf<error::box>(
					orig_me, a_load(), std::move(cur_fmt), std::move(cur_fname)
				);
			},

			// 3. patch 'a_save' request to be noop - until object will be actually read
			// => if object is not yet loaded, then saving to same file is noop
			[=](a_save, std::string cur_fmt, std::string cur_fname)
			mutable -> error::box {
				// noop if saving to same file with same format
				// otherwise invoke lazy load (read object) & then save it
				if(cur_fmt == fmt_name && cur_fname == fname)
					return success();
				else {
					// [NOTE] need `current_behavior()` because lazy load is noop in `orig_me`
					if(auto er = actorf<error::box>(current_behavior(), a_load()); er.ec)
						return er;
					// apply save job inplace
					// [NOTE] use `orig_me` here, because it have unpatched (normal) save
					return actorf<error::box>(
						orig_me, a_save(), std::move(cur_fmt), std::move(cur_fname)
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
objbase::~objbase() {
	// explicitly stop engine
	if(actor_)
		caf::anon_send_exit(actor_, caf::exit_reason::user_shutdown);
}

auto objbase::raw_actor() -> const caf::actor& {
	// engine must be initialized only once
	std::call_once(einit_flag_, [&] {
		// [NOTE] init may be called after move constructor, hence check if actor is initialized
		if(!actor_)
			actor_ = system().spawn_in_group<objbase_actor>(home(), shared_from_this());
	});
	return actor_;
}

auto objbase::home() const -> caf::group {
	return system().groups().get_local(home_id());
}

auto objbase::home_id() const -> std::string {
	return to_string(hid_);
}

auto objbase::apply(obj_transaction tr) -> tr_result {
	return actorf<tr_result>(
		actor(), kernel::radio::timeout(true), a_apply(), std::move(tr)
	);
}

auto objbase::apply(launch_async_t, obj_transaction tr) -> void {
	caf::anon_send(actor(), a_apply(), std::move(tr));
}

auto objbase::touch(tr_result tres) -> void {
	caf::anon_send(actor(), a_apply(), obj_transaction{
		[tres = std::move(tres)]() mutable { return std::move(tres); }
	});
}

NAMESPACE_END(blue_sky)
