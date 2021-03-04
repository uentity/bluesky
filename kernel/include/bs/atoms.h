/// @author uentity
/// @date 15.08.2018
/// @brief All atoms that are used in BS are declared here
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <caf/atom.hpp>

namespace blue_sky {
///////////////////////////////////////////////////////////////////////////////
//  common
//
/// denote that we don't want to wait until invoke result is available
using launch_async_t = caf::atom_constant<caf::atom("bs lasync")>;
inline constexpr auto launch_async = launch_async_t{};

/// denote operation that is thread-unsafe and can cause data race
using unsafe_t = caf::atom_constant<caf::atom("bs unsafe")>;
inline constexpr auto unsafe = unsafe_t{};

/// denote operation that can take long time
using long_op_t = caf::atom_constant<caf::atom("bs long op")>;
inline constexpr auto long_op = long_op_t{};

/// denote some operation that spans to whole subtree
using deep_t = caf::atom_constant<caf::atom("bs deep")>;
inline constexpr auto deep = deep_t{};

///////////////////////////////////////////////////////////////////////////////
//  generic BS API
//
// discover neighbourhood
using a_hi = caf::atom_constant<caf::atom("bs hi")>;
// used to inform others that I'm quit
using a_bye = caf::atom_constant<caf::atom("bs bye")>;
// used as 'acquired` tag
using a_ack = caf::atom_constant<caf::atom("bs ack")>;
// used to invoke some processing over an object/actor
using a_apply = caf::atom_constant<caf::atom("bs apply")>;
// indicate that operation is lazy (won't start immediately)
using a_lazy = caf::atom_constant<caf::atom("bs lazy")>;

// get implementation part of link/node/etc...
using a_impl = caf::atom_constant<caf::atom("bs impl")>;
// get home group of entity
using a_home = caf::atom_constant<caf::atom("bs home")>;
// get home group ID
using a_home_id = caf::atom_constant<caf::atom("bs home id")>;
// obtain data (retrive object)
using a_data = caf::atom_constant<caf::atom("bs data")>;
// obtain data node (retrive node)
using a_data_node = caf::atom_constant<caf::atom("bs dnode")>;
// object save/load from storage
using a_load = caf::atom_constant<caf::atom("bs load")>;
using a_save = caf::atom_constant<caf::atom("bs save")>;
// subscription manage
using a_subscribe = caf::atom_constant<caf::atom("bs subscrb")>;
// ask to clone some object
using a_clone = caf::atom_constant<caf::atom("bs clone")>;

///////////////////////////////////////////////////////////////////////////////
//  link API
//
using a_lnk_id = caf::atom_constant<caf::atom("tl id")>;
using a_lnk_name = caf::atom_constant<caf::atom("tl name")>;
using a_lnk_rename = caf::atom_constant<caf::atom("tl rename")>;

using a_lnk_status = caf::atom_constant<caf::atom("tl status")>;
using a_lnk_oid = caf::atom_constant<caf::atom("tl oid")>;
using a_lnk_otid = caf::atom_constant<caf::atom("tl otid")>;
using a_lnk_inode = caf::atom_constant<caf::atom("tl inode")>;
using a_lnk_flags = caf::atom_constant<caf::atom("tl flags")>;

using a_mlnk_fresh = caf::atom_constant<caf::atom("ml fresh")>;

// async invoke `fusion_link::populate()`
using a_flnk_data = caf::atom_constant<caf::atom("tfl data")>;
using a_flnk_populate = caf::atom_constant<caf::atom("tfl dnode")>;
using a_flnk_bridge = caf::atom_constant<caf::atom("tfl bridge")>;

///////////////////////////////////////////////////////////////////////////////
//  node API
//
using a_node_size = caf::atom_constant<caf::atom("tn size")>;
using a_node_leafs = caf::atom_constant<caf::atom("tn leafs")>;
using a_node_keys = caf::atom_constant<caf::atom("tn keys")>;
using a_node_ikeys = caf::atom_constant<caf::atom("tn ikeys")>;
using a_node_find = caf::atom_constant<caf::atom("tn find")>;
using a_node_index = caf::atom_constant<caf::atom("tn index")>;
using a_node_deep_search = caf::atom_constant<caf::atom("tn deeps")>;
using a_node_equal_range = caf::atom_constant<caf::atom("tn eqrng")>;
using a_node_deep_equal_range = caf::atom_constant<caf::atom("tn deqrng")>;
using a_node_insert = caf::atom_constant<caf::atom("tn insert")>;
using a_node_erase = caf::atom_constant<caf::atom("tn erase")>;
using a_node_clear = caf::atom_constant<caf::atom("tn clear")>;

// query node's actor group ID
using a_node_disconnect = caf::atom_constant<caf::atom("tn unplug")>;
using a_node_handle = caf::atom_constant<caf::atom("tn handle")>;
using a_node_rearrange = caf::atom_constant<caf::atom("tn rearng")>;

} /* namespace blue_sky */
