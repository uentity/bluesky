/// @file
/// @author uentity
/// @date 15.08.2018
/// @brief All atoms that are used in BS are declared here
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <caf/atom.hpp>

namespace blue_sky {
// async invoke `link::data()`
using lnk_data_atom = caf::atom_constant<caf::atom("tl data")>;
// async invoke `link::data_node()`
using lnk_dnode_atom = caf::atom_constant<caf::atom("tl dnode")>;
// async invoke `fusion_link::populate()`
using flnk_populate_atom = caf::atom_constant<caf::atom("tfl pull")>;
	
} /* namespace blue_sky */

