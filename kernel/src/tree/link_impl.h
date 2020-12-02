/// @file
/// @author uentity
/// @date 22.07.2019
/// @brief Link implementation that contain link state
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/actor_common.h>
#include <bs/objbase.h>
#include <bs/kernel/radio.h>
#include <bs/detail/function_view.h>
#include <bs/detail/sharded_mutex.h>
#include <bs/tree/node.h>

#include "private_common.h"
#include "engine_impl.h"
#include "../kernel/radio_subsyst.h"

#include <caf/actor.hpp>
#include <caf/typed_actor.hpp>

#include <variant>

#define LINK_TYPE_DEF(lnk_class, limpl_class, typename)                            \
ENGINE_TYPE_DEF(limpl_class, typename)                                             \
auto lnk_class::type_id_() -> std::string_view { return limpl_class::type_id_(); }

#define LINK_CONVERT_TO(lnk_class)                               \
lnk_class::lnk_class(const link& rhs) : link(rhs, type_id_()) {}

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  base link impl
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API link_impl :
	public engine::impl, public engine::impl::access<link>,
	public bs_detail::sharded_mutex<engine_impl_mutex>
{
public:
	using sp_limpl = std::shared_ptr<link_impl>;
	using sp_scoped_actor = engine::impl::sp_scoped_actor;

	///////////////////////////////////////////////////////////////////////////////
	//  private link messaging interface
	//
	using primary_actor_type = link::actor_type
	::extend_with<engine_actor_type<link>>
	::extend<
		// ask link's actor to send status changed ack AFTER it has been already changed by handle
		caf::reacts_to<a_lnk_status, Req, ReqStatus, ReqStatus>,
		// delayed read for cached links
		caf::replies_to<a_lazy, a_load, bool /* with_node */>::with<bool>
	>::extend_with<kernel::detail::khome_actor_type>;

	// ack signals that this link send to home group
	using self_ack_actor_type = caf::typed_actor<
		// ack rename
		caf::reacts_to<a_ack, a_lnk_rename, std::string, std::string>,
		// request status change ack
		caf::reacts_to<a_ack, a_lnk_status, Req, ReqStatus, ReqStatus>
	>;

	// home group receives only myself acks + (for hard link types) ack after object transaction
	using home_actor_type = self_ack_actor_type::extend<
		// data altered ack from object
		caf::reacts_to<a_ack, a_data, tr_result::box>
	>;

	// leaf acks coming from subtree
	using deep_ack_actor_type = caf::typed_actor<
		caf::reacts_to<a_ack, caf::actor, lid_type, a_lnk_rename, std::string, std::string>,
		caf::reacts_to<a_ack, caf::actor, lid_type, a_lnk_status, Req, ReqStatus, ReqStatus>,
		caf::reacts_to<a_ack, caf::actor, lid_type, a_data, tr_result::box>
	>;

	// foreign acks coming to link's home group from deeper levels
	using subtree_ack_actor_type = deep_ack_actor_type::extend<
		// node acks from deeper levels
		caf::reacts_to<a_ack, caf::actor, a_node_insert, lid_type, size_t>,
		caf::reacts_to<a_ack, caf::actor, a_node_insert, lid_type, size_t, size_t>,
		caf::reacts_to<a_ack, caf::actor, a_node_erase, lids_v>
	>;

	// all acks processed by link
	using ack_actor_type = self_ack_actor_type::extend_with<subtree_ack_actor_type>;

	// complete private actor type
	using actor_type = primary_actor_type::extend_with<ack_actor_type>;

	///////////////////////////////////////////////////////////////////////////////
	//  methods
	//
	//[NOTE] empty ctor required for serialization support (even in derived links)
	link_impl();
	link_impl(std::string name, Flags f);
	virtual ~link_impl();

	/// spawn raw actor corresponding to this impl type
	virtual auto spawn_actor(sp_limpl limpl) const -> caf::actor;

	/// clone this impl
	virtual auto clone(bool deep = false) const -> sp_limpl = 0;

	/// download pointee data
	virtual auto data() -> obj_or_err = 0;
	/// return cached pointee data (if any), default impl returns nullptr
	// [NOTE] overriden impl MUST NEVER start any resource-consuming task inside!
	virtual auto data(unsafe_t) const -> sp_obj;
	// links that doesn't operate with objbase (for ex, map_link), can override this
	// to provide direct access to stored node (default impl works via `data(unsafe)`)
	virtual auto data_node(unsafe_t) const -> node;

	/// obtain inode pointer
	/// default impl do it via `data_ex()` call
	virtual auto get_inode() -> result_or_err<inodeptr>;

	// if pointee is a node - set node's handle to self and return pointee
	// derived link can change default behaviour (for ex. if link cannot serve as node's unique handle)
	// [NOTE] unsafe -- operates directly on data
	virtual auto propagate_handle() -> node_or_err;

	// can be used by derived links to reset handle of given node
	auto propagate_handle(node& N) const -> node&;

	/// manipulate with owner (protected by mutex)
	auto owner() const -> node;
	auto reset_owner(const node& new_owner) -> void;

	auto req_status(Req request) const -> ReqStatus;

	// can return `false` to indicate failed postcondition & revert status to original value
	using on_rs_changed_fn = function_view< bool(Req, ReqStatus /*new*/, ReqStatus /*old*/) >;

	auto rs_reset(
		Req request, ReqReset cond, ReqStatus new_rs, ReqStatus old_rs, on_rs_changed_fn on_rs_changed
	) -> ReqStatus;

	// sends notification about status change to link's home group
	auto rs_reset(
		Req request, ReqReset cond, ReqStatus new_rs, ReqStatus old_rs = ReqStatus::Void
	) -> ReqStatus;

	// set proper status after request result was received & release pending waiters
	// [NOTE] only Broadcast flag is used from 2nd arg
	using req_result = std::variant<obj_or_errbox, node_or_errbox>;
	auto rs_update_from_data(req_result rdata, ReqReset opts = ReqReset::Always) -> void;

	// run generic function while holding exclusive lock to status handles
	// if ReqReset::Broadcast is on, lock both handles
	// returns request status value before call
	auto rs_apply(
		Req req, function_view< void() > f,
		ReqReset cond = ReqReset::Always, ReqStatus cond_value = ReqStatus::Void
	) -> ReqStatus;

	// add actor to list of waiters for request result
	auto rs_add_waiter(Req req, caf::actor w) -> void;

	// rename and send notification to home group
	auto rename(std::string new_name) -> void;

	/// create or set or create inode for given target object
	/// [NOTE] if `new_info` is non-null, returned inode may be NOT EQUAL to `new_info`
	static auto make_inode(const sp_obj& target, inodeptr new_info = nullptr) -> inodeptr;

	///////////////////////////////////////////////////////////////////////////////
	//  member variables
	//
	lid_type id_;
	std::string name_;
	Flags flags_;

	/// status of operations
	struct status_handle {
		ReqStatus value = ReqStatus::Void;
		mutable engine_impl_mutex guard;
		// list of waiters for request result (until value != Busy)
		std::vector<caf::actor> waiters;
	};

private:
	friend node_impl;

	// requests status
	status_handle status_[2];
	// owner node
	node::weak_ptr owner_;

	// direct access to status_handle
	auto req_status_handle(Req request) -> status_handle&;
};
using sp_limpl = link_impl::sp_limpl;

///////////////////////////////////////////////////////////////////////////////
//  derived links
//
struct BS_HIDDEN_API ilink_impl : link_impl {
	inodeptr inode_;

	using super = link_impl;

	ilink_impl();
	ilink_impl(std::string name, const sp_obj& data, Flags f);

	// returns stored pointer
	auto get_inode() -> result_or_err<inodeptr> override final;
};

struct BS_HIDDEN_API sym_link_impl : link_impl {
	std::string path_;

	using super = link_impl;

	sym_link_impl();
	sym_link_impl(std::string name, std::string path, Flags f);

	auto clone(bool deep = false) const -> sp_limpl override;

	auto spawn_actor(sp_limpl limpl) const -> caf::actor override;

	auto data() -> obj_or_err override;
	auto data(unsafe_t) const -> sp_obj override;

	auto target() const -> link_or_err;

	auto propagate_handle() -> node_or_err override;

	ENGINE_TYPE_DECL
};

auto to_string(Req) -> const char*;
auto to_string(ReqStatus) -> const char*;

NAMESPACE_END(blue_sky::tree)
