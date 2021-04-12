/// @date 21.09.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"

#define STD_HASH_BS_LINK(link_class)                                                     \
namespace std { template<> struct hash<::blue_sky::tree::link_class> {                   \
	auto operator()(const ::blue_sky::tree::link_class& L) const noexcept { return L.hash(); } \
}; }

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  link with minimal API that operates directly on internal data
 *-----------------------------------------------------------------------------*/
class BS_API bare_link {
public:
	/// bare link can only be obtained from `link` instance
	explicit bare_link(const link& rhs);

	/// assign from normal link
	auto operator =(const link& rhs) -> bare_link&;

	/// convert from bare to normal link
	auto armed() const -> link;

	/// test if link is nil
	auto is_nil() const -> bool;
	operator bool() const { return !is_nil(); }

	/// get link's container
	auto owner() const -> node;

	/// returns engine's string type ID
	auto type_id() const -> std::string_view;

	/// hash for appropriate containers
	auto hash() const noexcept -> std::size_t;

	/// swap spport
	friend auto swap(bare_link& lhs, bare_link& rhs) noexcept -> void {
		std::swap(lhs.pimpl_, rhs.pimpl_);
	}

	///////////////////////////////////////////////////////////////////////////////
	//  Main API with direct access to link internals
	//
	/// access link's unique ID
	auto id() const -> lid_type;

	/// obtain link's symbolic name
	auto name() const -> std::string;

	auto flags() const -> Flags;

	/// get link's object ID -- can return empty string
	auto oid() const -> std::string;

	/// get link's object type ID -- can return nil type ID
	auto obj_type_id() const -> std::string;

	/// obtain inode
	auto info() -> result_or_err<inode>;

	///////////////////////////////////////////////////////////////////////////////
	//  Pointee data API
	//
	/// get request status
	auto req_status(Req request) const -> ReqStatus;

	/// directly return cached value (if any)
	auto data() -> sp_obj;

	/// return node extracted from data_node(unsafe)
	auto data_node() -> node;
	/// atomically check status & return data_node(unsafe) only if status == OK
	auto data_node_if_ok() -> node;

	/// if pointee is a node, return node's actor group ID
	auto data_node_hid() -> std::string;

private:
	friend link;
	friend link_actor;

	std::shared_ptr<link_impl> pimpl_;

	explicit bare_link(std::shared_ptr<link_impl> impl);

	auto pimpl() const -> link_impl*;
};

NAMESPACE_END(blue_sky::tree)

STD_HASH_BS_LINK(bare_link)
