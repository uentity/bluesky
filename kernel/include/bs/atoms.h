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

// used as 'acquired` tag
using a_ack = caf::atom_constant<caf::atom("bs ack")>;
// used to inform others that I'm quit
using a_bye = caf::atom_constant<caf::atom("bs bye")>;
// signals that link or node ID changed
using a_bind_id = caf::atom_constant<caf::atom("bs bind id")>;

// async invoke `link::data()`
using a_lnk_data = caf::atom_constant<caf::atom("tl data")>;
// async invoke `link::data_node()`
using a_lnk_dnode = caf::atom_constant<caf::atom("tl dnode")>;
// async invoke `fusion_link::populate()`
using a_flnk_populate = caf::atom_constant<caf::atom("tfl pull")>;

using a_lnk_rename = caf::atom_constant<caf::atom("tl rename")>;
using a_lnk_insert = caf::atom_constant<caf::atom("tl insert")>;
using a_lnk_erase = caf::atom_constant<caf::atom("tl erase")>;

using a_lnk_status = caf::atom_constant<caf::atom("tl status")>;
using a_lnk_oid = caf::atom_constant<caf::atom("tl oid")>;
using a_lnk_otid = caf::atom_constant<caf::atom("tl otid")>;
using a_lnk_inode = caf::atom_constant<caf::atom("tl inode")>;

} /* namespace blue_sky */

