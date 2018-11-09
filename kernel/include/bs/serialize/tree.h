/// @file
/// @author uentity
/// @date 22.06.2018
/// @brief Declare serialization of BS tree related types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "../tree/tree.h"
#include "serialize.h"

// inode
BSS_FCN_DECL(save, blue_sky::tree::inode)
BSS_FCN_DECL(load, blue_sky::tree::inode)

// link
BSS_FCN_DECL(serialize, blue_sky::tree::link)

// ilink
BSS_FCN_DECL(serialize, blue_sky::tree::ilink)

// hard link
BSS_FCN_DECL(serialize, blue_sky::tree::hard_link)
BSS_FCN_DECL(load_and_construct, blue_sky::tree::hard_link)

// weak link
BSS_FCN_DECL(serialize, blue_sky::tree::weak_link)
BSS_FCN_DECL(load_and_construct, blue_sky::tree::weak_link)

// sym link
BSS_FCN_DECL(serialize, blue_sky::tree::sym_link)
BSS_FCN_DECL(load_and_construct, blue_sky::tree::sym_link)

// fusion link
BSS_FCN_DECL(serialize, blue_sky::tree::fusion_link)
BSS_FCN_DECL(load_and_construct, blue_sky::tree::fusion_link)

// node
BSS_FCN_DECL(serialize, blue_sky::tree::node)

BSS_FORCE_DYNAMIC_INIT(link)
BSS_FORCE_DYNAMIC_INIT(node)

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

/*-----------------------------------------------------------------------------
 *  Tree save/load to JSON archive
 *-----------------------------------------------------------------------------*/
auto save_tree(const sp_link& root, const std::string& filename) -> error;
auto load_tree(const std::string& filename) -> result_or_err<sp_link>;

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

