/// @file
/// @author uentity
/// @date 08.07.2019
/// @brief LInk's async actor
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/link.h>
#include <bs/tree/node.h>
#include <bs/kernel/radio.h>

#include "actor_common.h"
#include "link_impl.h"

#include <caf/actor_system.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
using namespace tree::detail;

using id_type = link::id_type;
using Flags = link::Flags;

/*-----------------------------------------------------------------------------
 *  link_actor
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API link_actor : public caf::event_based_actor {
public:
	//using super = caf::stateful_actor<lnk_state>;
	using super = caf::event_based_actor;
	using behavior_type = super::behavior_type;

	link_actor(caf::actor_config& cfg, sp_limpl Limpl);
	// virtual dtor
	virtual ~link_actor();

	// init new self group after ID has been set
	auto bind_new_id() -> void;
	// cleanup code executes leaving from local group & must be called from outside
	auto goodbye() -> void;

	// get handle of this actor
	inline auto handle() -> caf::actor {
		return caf::actor_cast<caf::actor>(address());
	}

	/// get pointer to object link is pointing to -- slow, never returns invalid (NULL) sp_obj
	virtual auto data_ex(bool wait_if_busy = true) -> result_or_err<sp_obj>;

	/// return tree::node if contained object is a node -- slow, never returns invalid (NULL) sp_obj
	virtual auto data_node_ex(bool wait_if_busy = true) -> result_or_err<sp_node>;

	///////////////////////////////////////////////////////////////////////////////
	//  behavior
	//
	// returns generic behavior
	auto make_behavior() -> behavior_type override;

	// Generic impl of link's beahvior suitable for all link types
	auto make_generic_behavior() -> behavior_type;

	auto on_exit() -> void override;

	auto name() const -> const char* override;

	/// raw implementation that don't manage status
	virtual auto data_node() -> result_or_err<sp_node>;

	// holds reference to link impl
	sp_limpl pimpl_;
	link_impl& impl;
};

/// installs behavior suitable for links that:
/// 1) contains direct ptr to object (data)
/// 2) access to object's data & node is fast, s.t. we don't need to manage req status
struct BS_HIDDEN_API simple_link_actor : public link_actor {
	using super = link_actor;
	using super::super;

	auto data_ex(bool wait_if_busy = true) -> result_or_err<sp_obj> override;
	auto data_node_ex(bool wait_if_busy = true) -> result_or_err<sp_node> override;
};

// helper for generating `link_impl::spawn_actor()` implementations
template<typename Actor, caf::spawn_options Os = caf::no_spawn_options, class... Ts>
inline auto spawn_lactor(std::shared_ptr<link_impl> limpl, Ts&&... args) {
	// spawn actor
	auto& AS = kernel::radio::system();
	auto lgrp = AS.groups().get_local(to_string(limpl->id_));
	return AS.spawn_in_group<Actor, Os>(
		std::move(lgrp), std::move(limpl), std::forward<Ts>(args)...
	);
}

NAMESPACE_END(blue_sky::tree)
