/// @file
/// @author uentity
/// @date 29.06.2018
/// @brief BS tree node implementation part of PIMPL
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/atoms.h>
#include <bs/actor_common.h>
#include <bs/tree/node.h>
#include <bs/kernel/radio.h>

#include "node_impl.h"

#include <boost/uuid/uuid_hash.hpp>

#include <caf/actor_cast.hpp>

#include <unordered_map>

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

	node_actor(caf::actor_config& cfg, sp_nimpl Nimpl);
	~node_actor();

	// return typed actor handle to this
	node_impl::actor_type handle() const {
		return caf::actor_cast<node_impl::actor_type>(address());
	}

	auto make_behavior() -> behavior_type override;

	auto on_exit() -> void override;

	auto name() const -> const char* override;

	// say goodbye to others & leave home
	auto goodbye() -> void;

	// get/set home group ID + optionally invite actor
	auto home() -> caf::group&;
	auto home(std::string gid) -> caf::group&;
	// 'unsafe' means read-only direct access to stored group memeber
	auto home(unsafe_t) const -> caf::group&;

	// get node's group ID
	auto gid() -> const std::string&;
	auto gid(unsafe_t) const -> std::string;

	auto insert(
		link L, const InsertPolicy pol, bool silent = false
	) -> insert_status<Key::ID>;

	auto insert(
		link L, std::size_t idx, const InsertPolicy pol, bool silent = false
	) -> node::insert_status;

	auto erase(const lid_type& key, EraseOpts opts = EraseOpts::Normal) -> size_t;

	auto retranslate_from(const link& L) -> void;
	auto stop_retranslate_from(const link& L) -> void;
	// stops retranslating from all leafs
	auto disconnect() -> void;
};

// helper for correct spawn of node actor
template<typename Actor = node_actor, caf::spawn_options Os = caf::no_spawn_options, class... Ts>
inline auto spawn_nactor(std::shared_ptr<node_impl> nimpl, caf::group nhome, Ts&&... args) {
	auto& AS = kernel::radio::system();
	if(nhome)
		return AS.spawn_in_group<Actor, Os>(nhome, std::move(nimpl), std::forward<Ts>(args)...);
	else // delayed self group creation
		return AS.spawn<Actor, Os>(std::move(nimpl), std::forward<Ts>(args)...);
}

NAMESPACE_END(blue_sky::tree)
