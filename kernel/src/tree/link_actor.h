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

#include "node_impl.h"

#include <caf/actor_system.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

enum class ReqOpts {
	WaitIfBusy = 0, ErrorIfBusy = 1, ErrorIfNOK = 2, Detached = 4, DirectInvoke = 8,
	HasDataCache = 16, Uniform = 32
};

/*-----------------------------------------------------------------------------
 *  link_actor
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API link_actor : public caf::event_based_actor {
public:
	using super = caf::event_based_actor;
	using primary_actor_type = link_impl::primary_actor_type;
	using ack_actor_type = link_impl::ack_actor_type;
	using actor_type = link_impl::actor_type;
	using typed_behavior = actor_type::behavior_type;
	using behavior_type = super::behavior_type;

	link_actor(caf::actor_config& cfg, caf::group self_grp, sp_limpl Limpl);
	// virtual dtor
	virtual ~link_actor();

	// cleanup code executes leaving from local group & must be called from outside
	auto goodbye() -> void;

	auto on_exit() -> void override;

	auto name() const -> const char* override;

	// get typed link actor handle
	inline auto actor() -> actor_type {
		return caf::actor_cast<actor_type>(this);
	}

	// forward call to impl + send notification to self group
	auto rs_reset(Req req, ReqReset cond, ReqStatus new_rs, ReqStatus prev_rs, bool silent = false)
	-> ReqStatus;

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

	// get link pointee, never returns invalid (NULL) sp_obj
	using obj_processor_f = std::function<  void(result_or_errbox<sp_obj>) >;
	virtual auto data_ex(obj_processor_f cb, ReqOpts opts) -> void;

	// return tree::node if contained object is a node, never returns nil node
	using node_processor_f = std::function< void(node_or_errbox) >;
	virtual auto data_node_ex(node_processor_f cb, ReqOpts opts) -> void;

	// parts of behavior
	auto make_primary_behavior() -> primary_actor_type::behavior_type;
	auto make_ack_behavior() -> ack_actor_type::behavior_type;
	// combined link behavior: primary + ack
	auto make_typed_behavior() -> typed_behavior;
	auto make_behavior() -> behavior_type override;

	// holds reference to link impl
	sp_limpl pimpl_;
	link_impl& impl;
};

// helper for generating `link_impl::spawn_actor()` implementations
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
	using super::super;

	// part of behavior overloaded by this actor
	// OID & obj type ID getters always applied to data cache
	using typed_behavior_overload = caf::typed_behavior<
		// get pointee OID
		caf::replies_to<a_lnk_oid>::with<std::string>,
		// get pointee type ID
		caf::replies_to<a_lnk_otid>::with<std::string>,
		// delayed object load
		caf::replies_to<a_delay_load>::with<bool>
	>;

	auto data_ex(obj_processor_f cb, ReqOpts opts) -> void override;
	auto data_node_ex(node_processor_f cb, ReqOpts opts) -> void override;

	auto make_typed_behavior() -> typed_behavior;
	auto make_behavior() -> behavior_type override;
};

NAMESPACE_END(blue_sky::tree)
