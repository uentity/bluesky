/// @file
/// @author uentity
/// @date 23.10.2018
/// @brief Inode (link metadata) implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/inode.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

// ctor
inode::inode(Flags f, std::uint8_t u_, std::uint8_t g_, std::uint8_t o_)
	: flags(f), u(u_), g(g_), o(o_), mod_time(make_timestamp())
{}

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

