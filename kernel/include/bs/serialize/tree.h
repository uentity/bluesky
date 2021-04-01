/// @file
/// @author uentity
/// @date 22.06.2018
/// @brief Declare serialization of BS tree related types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "serialize_decl.h"
#include "../tree/link.h"
#include "../tree/node.h"
#include "../tree/type_caf_id.h"

// inode
BSS_FCN_DECL(save, blue_sky::tree::inode)
BSS_FCN_DECL(load, blue_sky::tree::inode)

// link
BSS_FCN_DECL(serialize, blue_sky::tree::link)

// node
BSS_FCN_DECL(serialize, blue_sky::tree::node)

NAMESPACE_BEGIN(blue_sky::tree)

// derived links are serialized as base link
template<typename Archive, typename T>
auto serialize(Archive& ar, T& t, const std::uint32_t ver)
-> std::enable_if_t<std::is_base_of_v<link, T> && !std::is_same_v<T, link>, void> {
	serialize(ar, static_cast<link&>(t), ver);
}

NAMESPACE_END(blue_sky::tree)

BSS_FORCE_DYNAMIC_INIT(link)
BSS_FORCE_DYNAMIC_INIT(node)
