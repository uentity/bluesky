/// @file
/// @author uentity
/// @date 08.07.2019
/// @brief LInk's async actor
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/actor_common.h>
#include <bs/kernel/radio.h>

#include "engine_actor.h"
#include "node_impl.h"

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  link_actor
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API link_actor : public engine_actor<link> {
public:
	using super = engine_actor;
	using primary_actor_type = link_impl::primary_actor_type;
	using ack_actor_type = link_impl::ack_actor_type;
	using behavior_type = super::behavior_type;

	link_actor(caf::actor_config& cfg, caf::group home, sp_limpl Limpl);

	auto name() const -> const char* override;

	// pass message to upper (owner) level of tree structure
	template<typename... Args>
	auto forward_up(Args&&... args) -> void {
		// link forward messages directly to owner's home group
		if(auto master = impl.owner())
			checked_send<node_impl::home_actor_type, high_prio>(
				*this, master.home(), std::forward<Args>(args)...
			);
	}

	// forward 'ack' message to upper level, auto prepend it with this link ID info
	template<typename... Args>
	auto ack_up(Args&&... args) -> void {
		forward_up(a_ack(), impl.id_, std::forward<Args>(args)...);
	}

	// parts of behavior
	auto make_primary_behavior() -> primary_actor_type::behavior_type;
	auto make_ack_behavior() -> ack_actor_type::behavior_type;
	// complete & unboxed behavior
	auto make_typed_behavior() -> typed_behavior;
	auto make_behavior() -> behavior_type override;

protected:
	// default options for making Data & DataNode queries
	struct req_opts {
		ReqOpts data, data_node;
	};
	req_opts ropts_;
};

// spawns link actor inside link home group
template<typename Actor, caf::spawn_options Os = caf::no_spawn_options, class... Ts>
inline auto spawn_lactor(sp_limpl limpl, Ts&&... args) {
	// spawn actor
	auto& AS = kernel::radio::system();
	auto lgrp = AS.groups().get_local(to_string(limpl->id_));
	return AS.spawn_in_group<Actor, Os>(
		lgrp, lgrp, std::move(limpl), std::forward<Ts>(args)...
	);
}

/*-----------------------------------------------------------------------------
 *  customized actors for different links
 *-----------------------------------------------------------------------------*/
/// assumes that link contains cache for object's data
/// and if status is OK, it directly returns cached value
struct BS_HIDDEN_API cached_link_actor : public link_actor {
	using super = link_actor;
	using super::typed_behavior;

	cached_link_actor(caf::actor_config& cfg, caf::group self_grp, sp_limpl Limpl);

	// part of behavior overloaded by this actor
	// OID & obj type ID getters always applied to data cache
	using typed_behavior_overload = caf::typed_behavior<
		// get pointee OID
		caf::replies_to<a_lnk_oid>::with<std::string>,
		// get pointee type ID
		caf::replies_to<a_lnk_otid>::with<std::string>,
		// delayed object load
		caf::replies_to<a_lazy, a_load, bool /* with_node */>::with<bool>
	>;

	auto make_typed_behavior() -> typed_behavior;
	auto make_behavior() -> behavior_type override;
};

NAMESPACE_END(blue_sky::tree)
