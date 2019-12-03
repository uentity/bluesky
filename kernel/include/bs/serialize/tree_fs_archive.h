/// @file
/// @author uentity
/// @date 21.06.2019
/// @brief Header that unites tree FS save & load archives
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "tree_fs_input.h"
#include "tree_fs_output.h"

// tie input and output archives together
CEREAL_SETUP_ARCHIVE_TRAITS(blue_sky::tree_fs_input, blue_sky::tree_fs_output)

