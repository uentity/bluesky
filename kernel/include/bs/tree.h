/// @file
/// @author uentity
/// @date 22.11.2017
/// @brief blue-sky tree summary header
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "link.h"
#include "node.h"

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

/// returns absolute link's path consisting from link IDs separated by '/'
/// if `human_readable` is set, replace IDs with link names
BS_API std::string abspath(sp_clink l, bool human_readable = false);

/// convert path from ID-based to human readable (name-bases) or vice versa
BS_API std::string convert_path(std::string src_path, const sp_clink& start, bool from_human_readable = false);

/// quick link search by given abs path (ID-based!)
/// may be faster that full `node::deep_search()`
/// also can lookup starting from any tree node given absolute path
BS_API sp_link search_by_path(const std::string& path, const sp_clink& start);

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

