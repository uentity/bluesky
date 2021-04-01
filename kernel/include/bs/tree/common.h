/// @author uentity
/// @date 05.12.2019
/// @brief Common definitions for BS tree
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../common.h"
#include "../error.h"
#include "../uuid.h"
#include "../transaction.h"
#include "../detail/enumops.h"
#include "../detail/function_view.h"
#include "errors.h"
#include "inode.h"

#include <caf/actor.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
// denote possible tree events
enum class Event : std::uint32_t {
	None = 0,
	LinkRenamed = 1,
	LinkStatusChanged = 2,
	LinkInserted = 4,
	LinkErased = 8,
	LinkDeleted = 16,
	DataModified = 32,
	DataNodeModified = 4 + 8,
	All = std::uint32_t(-1)
};

/// link object data requests
enum class Req { Data = 0, DataNode = 1 };
/// request status reset conditions
enum class ReqReset { Always = 0, IfEq = 1, IfNeq = 2, Broadcast = 4 };
/// states of reuqest
enum class ReqStatus { Void, Busy, OK, Error };

/// flags reflect link properties and state
enum Flags : std::uint8_t {
	Plain = 0,
	Persistent = 1,
	Disabled = 2,
	LazyLoad = 4
};

/// node leafs indexes/ordering
enum class Key { ID, OID, Name, Type, AnyOrder };

/// links insertions policy
enum class InsertPolicy : std::uint8_t {
	AllowDupNames = 0,
	DenyDupNames = 1,
	RenameDup = 2,
	Merge = 4
};

/// options for algorithms working with tree
enum class TreeOpts : std::uint32_t {
	Normal = 0,
	WalkUp = 2,
	Deep = 4,
	Lazy = 8,
	FollowSymLinks = 16,
	FollowLazyLinks = 32,
	MuteOutputNode = 64,
	HighPriority = 256,
	DetachedWorkers = 512,
	TrackWorkers = 1024
};

/// link's unique ID type
using lid_type = uuid;

/// forward declare major types
class link;
class link_impl;
class link_actor;

class node;
class node_impl;
class node_actor;

using obj_or_err = result_or_err<sp_obj>;
using obj_or_errbox = result_or_errbox<sp_obj>;
using link_or_err = result_or_err<link>;
using link_or_errbox = result_or_errbox<link>;
using node_or_err = result_or_err<node>;
using node_or_errbox = result_or_errbox<node>;

struct event {
	caf::actor origin;
	prop::propdict params;
	Event code;

	auto origin_link() const -> link;
	auto origin_node() const -> node;
};

NAMESPACE_END(blue_sky::tree)

BS_ALLOW_ENUMOPS(tree::Event)
BS_ALLOW_ENUMOPS(tree::ReqReset)
BS_ALLOW_ENUMOPS(tree::Flags)
BS_ALLOW_ENUMOPS(tree::InsertPolicy)
BS_ALLOW_ENUMOPS(tree::TreeOpts)
