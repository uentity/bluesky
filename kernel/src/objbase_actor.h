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
		// sent to home when owner exists
		caf::reacts_to<a_bye>,
		// modification ack
		caf::reacts_to<a_ack, a_data, tr_result::box>
	>;

	using actor_type = objbase::actor_type::extend<
		// runs async transaction
		caf::replies_to<a_ack, a_apply, obj_transaction>::with<tr_result::box>,
		// invoke object save
		caf::replies_to<a_save, std::string /* fmt */, std::string /* fname */>::with<error::box>,
		// invoke object load
		caf::replies_to<a_load, std::string /* fmt */, std::string /* fname */>::with<error::box>,
		// setup lazy read action
		caf::replies_to<
			a_lazy, a_load, std::string /* fmt */, std::string /* fname */, bool /* read node from file? */
		>::with<bool>,
		// if lazy load was set up, tells whether node will be read from file
		caf::replies_to<a_lazy, a_load, a_data_node>::with<bool>,
		// trigger lazy load
		caf::replies_to<a_load>::with<error::box>
	>
	::extend_with<home_actor_type>;

	using typed_behavior = actor_type::behavior_type;

	static auto actor(objbase& obj) {
		return caf::actor_cast<actor_type>(obj.actor());
	}

	objbase_actor(caf::actor_config& cfg, sp_obj mama);

	auto name() const -> const char* override;

	auto make_typed_behavior() -> typed_behavior;
	auto make_behavior() -> behavior_type override;

	auto on_exit() -> void override;

	///////////////////////////////////////////////////////////////////////////////
	//  member variables
	//
	caf::group home_;
	std::weak_ptr<objbase> mama_;
};

NAMESPACE_END(blue_sky)
