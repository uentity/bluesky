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
#include <bs/defaults.h>
#include <bs/objbase.h>
#include <bs/kernel/radio.h>
#include <bs/detail/enumops.h>
#include <bs/detail/function_view.h>
#include <bs/detail/sharded_mutex.h>
#include <bs/tree/node.h>

#include "engine_impl.h"
#include "../kernel/radio_subsyst.h"

#include <caf/detail/shared_spinlock.hpp>

#define LINK_TYPE_DEF(lnk_class, limpl_class, typename)                            \
ENGINE_TYPE_DEF(limpl_class, typename)                                             \
auto lnk_class::type_id_() -> std::string_view { return limpl_class::type_id_(); }

#define LINK_CONVERT_TO(lnk_class)                               \
lnk_class::lnk_class(const link& rhs) : link(rhs, type_id_()) {}

NAMESPACE_BEGIN(blue_sky::tree)
namespace bs_detail = blue_sky::detail;

using defaults::tree::nil_uid;
using defaults::tree::nil_oid;
inline const auto nil_otid = blue_sky::defaults::nil_type_name;

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
	using primary_actor_type = link::actor_type::extend<
		// run transaction in link's queue
		caf::replies_to<a_apply, simple_transaction>::with<error::box>,
		caf::replies_to<a_apply, link_transaction>::with<error::box>,
		// ask link's actor to send status changed ack AFTER it has been already changed by handle
		caf::reacts_to<a_lnk_status, Req, ReqStatus, ReqStatus>,
		// obtain link impl
		caf::replies_to<a_impl>::with<sp_limpl>,
		// delayed read for cached links
		caf::replies_to<a_delay_load>::with<bool>
	>::extend_with<kernel::detail::khome_actor_type>;

	// ack signals that this link send to home group
	using self_ack_actor_type = caf::typed_actor<
		// ack rename
		caf::reacts_to<a_ack, a_lnk_rename, std::string, std::string>,
		// request status change ack
		caf::reacts_to<a_ack, a_lnk_status, Req, ReqStatus, ReqStatus>,
		// data altered ack from object
		caf::reacts_to<a_ack, a_data, tr_result::box>
	>;

	// home group receives only myself acks
	using home_actor_type = self_ack_actor_type;

	// foreign acks coming to link's home group from deeper levels
	using subtree_ack_actor_type = caf::typed_actor<
		// leafs changes on deeper levels
		caf::reacts_to<a_ack, caf::actor, lid_type, a_lnk_rename, std::string, std::string>,
		caf::reacts_to<a_ack, caf::actor, lid_type, a_lnk_status, Req, ReqStatus, ReqStatus>,
		caf::reacts_to<a_ack, caf::actor, lid_type, a_data, tr_result::box>,

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

	/// obtain inode pointer
	/// default impl do it via `data_ex()` call
	virtual auto get_inode() -> result_or_err<inodeptr>;

	// if pointee is a node - set node's handle to self and return pointee
	// derived link can change default behaviour (for ex. if link cannot serve as node's unique handle)
	// [NOTE] unsafe -- operates directly on data
	virtual auto propagate_handle() -> node_or_err;

	/// manipulate with owner (protected by mutex)
	auto owner() const -> node;
	auto reset_owner(const node& new_owner) -> void;

	auto req_status(Req request) const -> ReqStatus;

	using on_rs_changed_fn = function_view< void(Req, ReqStatus /*new*/, ReqStatus /*old*/) >;

	auto rs_reset(
		Req request, ReqReset cond, ReqStatus new_rs, ReqStatus old_rs = ReqStatus::Void,
		on_rs_changed_fn on_rs_changed = noop
	) -> ReqStatus;

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
		// [NOTE] waiter returns type-erased pointer to captured result processor
		std::vector<std::function< void*() >> waiters;
	};

	// direct access to status_handle
	auto req_status_handle(Req request) -> status_handle&;

private:
	friend node_impl;

	// requests status
	status_handle status_[2];
	// owner node
	node::weak_ptr owner_;
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
