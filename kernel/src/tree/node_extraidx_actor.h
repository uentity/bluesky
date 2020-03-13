/// @file
/// @author uentity
/// @date 29.01.2020
/// @brief Support for OID and object type ID node indexes
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "actor_common.h"
#include "node_impl.h"

NAMESPACE_BEGIN(blue_sky::tree)

// search API - fully stateless, operates on incoming leafs vector
using extraidx_search_api = caf::typed_actor<
	// sort content according to given index
	caf::replies_to<a_node_leafs, Key, links_v>::with<links_v>,
	// find link by string key & meaning
	caf::replies_to<a_node_find, std::string, Key, links_v>::with<link>,
	// return index of link with string key & meaning
	caf::replies_to<a_node_index, std::string, Key, links_v>::with<node::existing_index>,
	// equal_range
	caf::replies_to<a_node_equal_range, std::string, Key, links_v>::with<links_v>
>;

auto extraidx_search_actor(extraidx_search_api::pointer) -> extraidx_search_api::behavior_type;

// deep search - requires node actor
using extraidx_deep_search_api = caf::typed_actor<
	caf::replies_to<a_node_deep_search, lid_type>::with<link>,
	caf::replies_to<a_node_deep_search, std::string, Key, bool>::with<links_v>
>;

auto extraidx_deep_search_actor(extraidx_deep_search_api::pointer, node_impl::actor_type Nactor)
-> extraidx_deep_search_api::behavior_type;

// erase - requires node actor
using extraidx_erase_api = caf::typed_actor<
	caf::replies_to<a_node_erase, std::string, Key, links_v>::with<std::size_t>
>;

// actor impl
auto extraidx_erase_actor(extraidx_erase_api::pointer, node_impl::actor_type Nactor)
-> extraidx_erase_api::behavior_type;

NAMESPACE_END(blue_sky::tree)
