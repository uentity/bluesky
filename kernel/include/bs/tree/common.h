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
#include "../detail/enumops.h"
#include "../detail/function_view.h"
#include "errors.h"
#include "inode.h"

#include <caf/allowed_unsafe_message_type.hpp>

NAMESPACE_BEGIN(blue_sky)

/// transaction is a function that is executed atomically in actor handler of corresponding object
template<typename T> using transaction_t = std::function< error(T) >;
using transaction = std::function< error() >;
using obj_transaction =  transaction_t<sp_obj>;
using link_transaction = transaction_t<tree::link>;
using node_transaction = transaction_t<tree::node>;

NAMESPACE_BEGIN(tree)
// denote possible tree events
enum class Event : std::uint32_t {
	LinkRenamed = 1,
	LinkStatusChanged = 2,
	LinkInserted = 4,
	LinkErased = 8,
	LinkDeleted = 16,
	All = std::uint32_t(-1)
};

/// link object data requests
enum class Req { Data = 0, DataNode = 1 };
/// request status reset conditions
enum class ReqReset {
	Always = 0, IfEq = 1, IfNeq = 2
};
/// states of reuqest
enum class ReqStatus { Void, Busy, OK, Error };

/// flags reflect link properties and state
enum Flags {
	Plain = 0,
	Persistent = 1,
	Disabled = 2,
	LazyLoad = 4
};

/// node leafs indexes/ordering
enum class Key { ID, OID, Name, Type, AnyOrder };

/// links insertions policy
enum class InsertPolicy {
	AllowDupNames = 0,
	DenyDupNames = 1,
	RenameDup = 2,
	Merge = 4
};

/// options for algorithms working with tree
enum class TreeOpts : unsigned {
	Normal = 0,
	WalkUp = 2,
	Deep = 4,
	Lazy = 8,
	FollowSymLinks = 16,
	FollowLazyLinks = 32,
	HighPriority = 256
};

/// link's unique ID type
using lid_type = uuid;

/// can be passed as callback that does nothing
inline constexpr auto noop = [](auto&&...) {};

template<typename R>
constexpr auto noop_r(R res = {}) {
	return [res = std::move(res)](auto&&...) mutable -> R { return res; };
}

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

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

// mark transaction as non-serializable
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::transaction)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::obj_transaction)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::link_transaction)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::node_transaction)
