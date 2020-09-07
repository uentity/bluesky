/// @author Alexander Gagarin (@uentity)
/// @date 22.07.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/objbase.h>
#include <bs/actor_common.h>

#include "kernel/radio_subsyst.h"

NAMESPACE_BEGIN(blue_sky)

class BS_HIDDEN_API objbase_actor : public caf::event_based_actor {
public:
	using super = caf::event_based_actor;
	using behavior_type = super::behavior_type;

	using home_actor_type = caf::typed_actor<
		// reset home group
		caf::reacts_to<a_home, std::string>,
		// modification ack
		caf::reacts_to<a_ack, a_data, tr_result::box>
	>;

	using actor_type = objbase::actor_type::extend<
			// setup delay read action
			caf::replies_to<a_delay_load, std::string /* fmt */, std::string /* fname */>::with<bool>,
	   		// trigger delay read
			caf::replies_to<a_delay_load, sp_obj /* obj */>::with<error::box>
		>
		::extend_with<kernel::detail::khome_actor_type>
		::extend_with<home_actor_type>
	;
	using typed_behavior = actor_type::behavior_type;

	caf::group home_;

	static auto actor(const objbase& obj) {
		return caf::actor_cast<actor_type>(obj.actor());
	}

	objbase_actor(caf::actor_config& cfg, caf::group home);

	auto make_typed_behavior() -> typed_behavior;
	auto make_behavior() -> behavior_type override;

	auto on_exit() -> void override;
};

NAMESPACE_END(blue_sky)
