/// @file
/// @author uentity
/// @date 29.06.2018
/// @brief BS tree node implementation part of PIMPL
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/node.h>
#include <bs/kernel/radio.h>

#include "actor_common.h"
#include "node_impl.h"

#include <caf/actor_cast.hpp>

#include <unordered_map>

#include <boost/uuid/uuid_hash.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;

/*-----------------------------------------------------------------------------
 *  node_actor
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API node_actor : public caf::event_based_actor {
public:
	using super = caf::event_based_actor;

	// holds reference to node impl
	sp_nimpl pimpl_;
	node_impl& impl;

	// map links to retranslator actors
	using axon_t = std::pair<std::uint64_t, std::optional<std::uint64_t>>;
	std::unordered_map<lid_type, axon_t> axons_;

	node_actor(caf::actor_config& cfg, caf::group ngrp, sp_nimpl Nimpl);
	virtual ~node_actor();

	// return typed actor handle to this
	node_impl::actor_type handle() const {
		return caf::actor_cast<node_impl::actor_type>(address());
	}

	// say goodbye to others & leave self group
	auto goodbye() -> void;

	auto name() const -> const char* override;
	auto make_behavior() -> behavior_type override;

	auto insert(
		sp_link L, const InsertPolicy pol, bool silent = false
	) -> insert_status<Key::ID>;

	auto insert(
		sp_link L, std::size_t idx, const InsertPolicy pol, bool silent = false
	) -> node::insert_status;

	auto erase(const lid_type& key, EraseOpts opts = EraseOpts::Normal) -> size_t;

	auto retranslate_from(const sp_link& L) -> void;
	auto stop_retranslate_from(const sp_link& L) -> void;
	// stops retranslating from all leafs
	auto disconnect() -> void;
};

// helper for correct spawn of node actor
template<typename Actor = node_actor, caf::spawn_options Os = caf::no_spawn_options, class... Ts>
inline auto spawn_nactor(std::shared_ptr<node_impl> nimpl, const std::string& gid, Ts&&... args) {
	auto& AS = kernel::radio::system();
	// create unique UUID for node's group because object IDs can be non-unique
	auto ngrp = AS.groups().get_local(gid);
	return AS.spawn_in_group<Actor, Os>(
		ngrp, ngrp, std::move(nimpl), std::forward<Ts>(args)...
	);
}

NAMESPACE_END(blue_sky::tree)
