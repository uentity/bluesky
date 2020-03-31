/// @file
/// @author uentity
/// @date 05.12.2019
/// @brief Common definitions for BS tree
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../common.h"
#include "../detail/enumops.h"
#include "../detail/function_view.h"
#include "inode.h"

#include <boost/uuid/uuid.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

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

/// link's unique ID type
using lid_type = boost::uuids::uuid;
/// function that modifies link's pointee
using data_modificator_f = objbase::modificator_f;

/// convert string to lid_type eating exceptions
/// decode error will be logged
BS_API auto to_uuid(std::string_view s) noexcept -> std::optional<lid_type>;

/// apply given function to successfully decoded input, return `defval` as fallback value
template<typename F>
auto if_uuid(std::string_view s, F&& f, std::invoke_result_t<F, lid_type> defval)
noexcept(noexcept( f(std::declval<lid_type>()) )) {
	if(auto u = to_uuid(s)) return f(*u);
	return defval;
}
/// overload for void
inline auto if_uuid(std::string_view s, function_view< void (lid_type) > f)
noexcept(noexcept( f(std::declval<lid_type>()) )) -> void {
	if(auto u = to_uuid(s)) f(*u);
}

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

NAMESPACE_END(blue_sky::tree)
