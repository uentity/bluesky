/// @file
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

// discover neighbourhood
using a_hi = caf::atom_constant<caf::atom("bs hi")>;
// used to inform others that I'm quit
using a_bye = caf::atom_constant<caf::atom("bs bye")>;
// used as 'acquired` tag
using a_ack = caf::atom_constant<caf::atom("bs ack")>;
// used to invoke some processing over an object/actor
using a_apply = caf::atom_constant<caf::atom("bs apply")>;

// async invoke `link::data()`
using a_lnk_data = caf::atom_constant<caf::atom("tl data")>;
using a_lnk_dcache = caf::atom_constant<caf::atom("tl dcache")>;
// async invoke `link::data_node()`
using a_lnk_dnode = caf::atom_constant<caf::atom("tl dnode")>;
// async invoke `fusion_link::populate()`
using a_flnk_populate = caf::atom_constant<caf::atom("tfl pull")>;
using a_flnk_bridge = caf::atom_constant<caf::atom("tfl bridge")>;

using a_lnk_id = caf::atom_constant<caf::atom("tl id")>;
using a_lnk_name = caf::atom_constant<caf::atom("tl name")>;
using a_lnk_rename = caf::atom_constant<caf::atom("tl rename")>;
using a_lnk_insert = caf::atom_constant<caf::atom("tl insert")>;
using a_lnk_erase = caf::atom_constant<caf::atom("tl erase")>;
using a_lnk_find = caf::atom_constant<caf::atom("tl find")>;

using a_lnk_status = caf::atom_constant<caf::atom("tl status")>;
using a_lnk_oid = caf::atom_constant<caf::atom("tl oid")>;
using a_lnk_otid = caf::atom_constant<caf::atom("tl otid")>;
using a_lnk_inode = caf::atom_constant<caf::atom("tl inode")>;
using a_lnk_flags = caf::atom_constant<caf::atom("tl flags")>;

// query node's actor group ID
using a_node_gid = caf::atom_constant<caf::atom("tn gid")>;
using a_node_disconnect = caf::atom_constant<caf::atom("tn unplug")>;
using a_node_propagate_owner = caf::atom_constant<caf::atom("tn powner")>;
using a_node_handle = caf::atom_constant<caf::atom("tn handle")>;
using a_node_size = caf::atom_constant<caf::atom("tn size")>;
using a_node_rearrange = caf::atom_constant<caf::atom("tn rearng")>;
using a_node_leafs = caf::atom_constant<caf::atom("tn leafs")>;

} /* namespace blue_sky */

