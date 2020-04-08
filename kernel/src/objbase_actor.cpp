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
#include <bs/kernel/config.h>
#include <bs/kernel/radio.h>
#include <bs/serialize/cafbind.h>

NAMESPACE_BEGIN(blue_sky)
using namespace kernel::radio;
using namespace std::chrono_literals;

using modificator_f = objbase::modificator_f;
using closed_modificator_f = objbase::closed_modificator_f;

NAMESPACE_BEGIN()

struct objbase_actor : public caf::event_based_actor {
	using actor_type = objbase::actor_type;
	using typed_behavior = actor_type::behavior_type;
	using super = caf::event_based_actor;
	using behavior_type = super::behavior_type;

	using modificator_f = objbase::modificator_f;

	objbase_actor(caf::actor_config& cfg) :
		super(cfg)
	{}

	auto make_typed_behavior() -> typed_behavior {
	return typed_behavior {

		// execute modificator
		[=](a_apply, const closed_modificator_f& m) mutable -> error::box {
			// invoke modificator
			return error::eval_safe([&]{ return m(); });
		}

	}; }

	auto make_behavior() -> behavior_type override {
		return make_typed_behavior().unbox();
	}
};

NAMESPACE_END()

auto objbase::start_engine() -> bool {
	if(!actor_) {
		actor_ = kernel::radio::system().spawn<objbase_actor>();
		return true;
	}
	return false;
}

auto objbase::apply(modificator_f m) const -> error {
	return actorf<error>(
		actor(), def_timeout(true), a_apply(), make_closed_modificator(std::move(m))
	);
}

auto objbase::apply(launch_async_t, modificator_f m) const -> void {
	caf::anon_send(actor(), a_apply(), make_closed_modificator(std::move(m)));
}

NAMESPACE_END(blue_sky)
