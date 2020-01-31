/// @file
/// @author uentity
/// @date 30.01.2020
/// @brief Messages retranslators for tree life support
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "node_actor.h"
#include <bs/tree/common.h>
#include <bs/tree/link.h>
#include <bs/log.h>

#include <caf/group.hpp>
#include <caf/stateful_actor.hpp>

// [TODO] refactor & remove debug support from here
#ifndef DEBUG_ACTOR
	#define DEBUG_ACTOR 0
#endif

#if DEBUG_ACTOR == 1
#include <caf/actor_ostream.hpp>
#endif

NAMESPACE_BEGIN()
#if DEBUG_ACTOR == 1

template<typename Actor>
auto adbg(Actor* A, const std::string& nid = {}) -> caf::actor_ostream {
	auto res = caf::aout(A);
	if constexpr(std::is_same_v<Actor, node_actor>) {
		res << "[N] ";
		if(auto pgrp = A->impl.self_grp.get())
			res << "[" << A->impl.self_grp.get()->identifier() << "]";
		else
			res << "[null grp]";
		res <<  ": ";
	}
	else if(!nid.empty() && nid.front() != '"') {
		res << "[N] [" << nid << "]: ";
	}
	return res;
}

#else

template<typename Actor>
constexpr auto adbg(const Actor*, const std::string& = {}) {
	return blue_sky::log::D();
}

#endif
NAMESPACE_END()

NAMESPACE_BEGIN(blue_sky::tree)

// state for node & link retranslators
struct rsl_state {
	lid_type src_lid;
	caf::group src_grp;
	caf::group tgt_grp;

	auto src_grp_id() const -> const std::string& {
		return src_grp ? src_grp.get()->identifier() : nil_grp_id;
	}
	auto tgt_grp_id() const -> const std::string& {
		return tgt_grp ? tgt_grp.get()->identifier() : nil_grp_id;
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
