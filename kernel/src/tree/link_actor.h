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

using id_type = link::id_type;
using Flags = link::Flags;

enum class ReqOpts {
	WaitIfBusy = 0, ErrorIfBusy = 1, ErrorIfNOK = 2, Detached = 4, DirectInvoke = 8,
	HasDataCache = 16
};

/*-----------------------------------------------------------------------------
 *  link_actor
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API link_actor : public caf::event_based_actor {
public:
	//using super = caf::stateful_actor<lnk_state>;
	using super = caf::event_based_actor;
	using behavior_type = super::behavior_type;

	link_actor(caf::actor_config& cfg, caf::group self_grp, sp_limpl Limpl);
	// virtual dtor
	virtual ~link_actor();

	// cleanup code executes leaving from local group & must be called from outside
	auto goodbye() -> void;

	// get handle of this actor
	inline auto handle() -> caf::actor {
		return caf::actor_cast<caf::actor>(address());
	}

	/// get pointer to object link is pointing to, never returns invalid (NULL) sp_obj
	using obj_processor_f = std::function<  void(result_or_errbox<sp_obj>) >;
	virtual auto data_ex(obj_processor_f cb, ReqOpts opts) -> void;

	/// return tree::node if contained object is a node, never returns invalid (NULL) sp_obj
	using node_processor_f = std::function< void(result_or_errbox<sp_node>) >;
	virtual auto data_node_ex(node_processor_f cb, ReqOpts opts) -> void;

	auto rename(std::string new_name, bool silent = false) -> void;

	// returns generic link behavior
	auto make_behavior() -> behavior_type override;

	auto on_exit() -> void override;

	auto name() const -> const char* override;

	// holds reference to link impl
	sp_limpl pimpl_;
	link_impl& impl;
};

/// assumes that link contains cache for object's data
/// and if status is OK, it directly returns cached value
struct BS_HIDDEN_API cached_link_actor : public link_actor {
	using super = link_actor;
	using super::super;

	auto data_ex(obj_processor_f cb, ReqOpts opts) -> void override;
	auto data_node_ex(node_processor_f cb, ReqOpts opts) -> void override;
};

/// 1) contains direct ptr to object (data)
/// 2) access to object's data & node is always fast, s.t. we don't need to manage req status
struct BS_HIDDEN_API fast_link_actor : public link_actor {
	using super = link_actor;
	using super::super;

	auto data_ex(obj_processor_f cb, ReqOpts opts) -> void override;
	auto data_node_ex(node_processor_f cb, ReqOpts opts) -> void override;

	auto make_behavior() -> behavior_type override;
};

// helper for generating `link_impl::spawn_actor()` implementations
template<typename Actor, caf::spawn_options Os = caf::no_spawn_options, class... Ts>
inline auto spawn_lactor(std::shared_ptr<link_impl> limpl, Ts&&... args) {
	// spawn actor
	auto& AS = kernel::radio::system();
	auto lgrp = AS.groups().get_local(to_string(limpl->id_));
	return AS.spawn_in_group<Actor, Os>(
		lgrp, lgrp, std::move(limpl), std::forward<Ts>(args)...
	);
}

NAMESPACE_END(blue_sky::tree)

BS_ALLOW_ENUMOPS(blue_sky::tree::ReqOpts)
