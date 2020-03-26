/// @file
/// @author uentity
/// @date 30.01.2020
/// @brief Messages retranslators for tree life support
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/defaults.h>
#include <bs/tree/common.h>
#include <bs/tree/link.h>
#include <bs/log.h>
#include "node_actor.h"

#include <caf/group.hpp>
#include <caf/stateful_actor.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

// state for node & link retranslators
struct rsl_state {
	lid_type src_lid;
	caf::group src_grp;
	caf::group tgt_grp;

	auto src_grp_id() const -> std::string_view {
		return src_grp ? src_grp.get()->identifier() : defaults::tree::nil_grp_id;
	}
	auto tgt_grp_id() const -> std::string_view {
		return tgt_grp ? tgt_grp.get()->identifier() : defaults::tree::nil_grp_id;
	}
};

struct node_rsl_state : rsl_state {
	link::actor_type src_actor;
};

// Link <-> Node retranslator
auto link_retranslator(caf::stateful_actor<rsl_state>* self, caf::group node_grp, lid_type lid)
-> caf::behavior;

// Node <-> Node retranslator
auto node_retranslator(
	caf::stateful_actor<node_rsl_state>* self, caf::group node_grp, lid_type lid, link::actor_type Lactor
) -> caf::behavior;

NAMESPACE_END(blue_sky::tree)
