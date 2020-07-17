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
#include <bs/serialize/cafbind.h>

#include "kernel/radio_subsyst.h"

NAMESPACE_BEGIN(blue_sky)
using namespace kernel::radio;
using namespace std::chrono_literals;

using modificator_f = objbase::modificator_f;
using closed_modificator_f = objbase::closed_modificator_f;

NAMESPACE_BEGIN()

class objbase_actor : public caf::event_based_actor {
public:
	using super = caf::event_based_actor;
	using behavior_type = super::behavior_type;

	using home_actor_type = caf::typed_actor<
		// reset home group
		caf::reacts_to<a_home, std::string>,
		// modification ack
		caf::reacts_to<a_ack, a_lnk_status, tree::ReqStatus>
	>;

	using actor_type = objbase::actor_type
		::extend_with<kernel::detail::khome_actor_type>
		::extend_with<home_actor_type>
	;
	using typed_behavior = actor_type::behavior_type;

	using modificator_f = objbase::modificator_f;

	caf::group home_;

	static auto actor(const objbase& obj) {
		return caf::actor_cast<actor_type>(obj.actor());
	}

	objbase_actor(caf::actor_config& cfg, caf::group home) :
		super(cfg), home_(std::move(home))
	{
		// exit after kernel
		KRADIO.register_citizen(this);
	}

	auto make_typed_behavior() -> typed_behavior {
	return typed_behavior {
		[=](a_bye) { if(current_sender() != this) quit(); },

		// get home group
		[=](a_home) { return home_; },

		[=](a_home, const std::string& new_hid) {
			leave(home_);
			send(home_, a_bye());
			join(system().groups().get_local(new_hid));
		},

		// execute modificator
		[=](a_apply, const closed_modificator_f& m) -> error::box {
			// invoke modificator
			auto er = error::eval_safe(m);
			auto s = er.ok() ? tree::ReqStatus::OK : tree::ReqStatus::Error;
			send(home_, a_ack(), a_lnk_status(), s);
			return er;
		},

		// skip acks - sent by myself
		[=](a_ack, a_lnk_status, tree::ReqStatus) {}

	}; }

	auto make_behavior() -> behavior_type override {
		return make_typed_behavior().unbox();
	}

	auto on_exit() -> void override {
		// say bye-bye to self group
		send(home_, a_bye());

		KRADIO.release_citizen(this);
	}
};

NAMESPACE_END()

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

auto objbase::home_id() const -> std::string {
	return home_ ? home_.get()->identifier() : "";
}

auto objbase::reset_home(std::string new_hid, bool silent) -> void {
	if(new_hid.empty()) new_hid = to_string(gen_uuid());
	home_ = system().groups().get_local(new_hid);
	if(!silent)
		caf::anon_send<high_prio>(objbase_actor::actor(*this), a_home(), std::move(new_hid));
}

auto objbase::apply(modificator_f m) const -> error {
	return actorf<error>(
		actor(), kernel::radio::timeout(true), a_apply(), make_closed_modificator(std::move(m))
	);
}

auto objbase::apply(launch_async_t, modificator_f m) const -> void {
	caf::anon_send(actor(), a_apply(), make_closed_modificator(std::move(m)));
}

NAMESPACE_END(blue_sky)
