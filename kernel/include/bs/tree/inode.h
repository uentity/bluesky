/// @file
/// @author uentity
/// @date 23.10.2018
/// @brief Inode (link metadata) structure definition
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once
#include "../error.h"
#include "../objbase.h"
#include "../detail/enumops.h"

#include <caf/timestamp.hpp>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

// time point type for all timestamps
using timestamp = caf::timestamp;
using timespan = caf::timespan;

/*-----------------------------------------------------------------------------
 s inode that stores access rights, timestampts, etc
 *-----------------------------------------------------------------------------*/
struct BS_API inode {
	/// flags reflect object properties and state
	enum Flags {
		Plain = 0
	};

	// 20-bit flags field + 12 bit = 4 bytes
	Flags flags : 20;
	// access rights
	// user (owner)
	bool : 1;
	std::uint8_t u : 3;
	// group
	bool : 1;
	std::uint8_t g : 3;
	// others
	bool : 1;
	std::uint8_t o : 3;

	// modification time
	timestamp mod_time;
	// link's owner
	std::string owner;
	std::string group;

	// do std initialization of all values
	inode(Flags f = Plain, std::uint8_t u = 7, std::uint8_t g = 5, std::uint8_t o = 5);
};

using inodeptr = std::shared_ptr<inode>;

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

