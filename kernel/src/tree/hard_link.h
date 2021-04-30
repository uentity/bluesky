/// @date 09.09.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "link_actor.h"
#include "../objbase_actor.h"

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  hard & weak links impl
 *-----------------------------------------------------------------------------*/
struct BS_HIDDEN_API hard_link_impl : ilink_impl {
	sp_obj data_;

	using super = ilink_impl;

	hard_link_impl();
	hard_link_impl(std::string name, sp_obj data, Flags f);

	auto clone(link_actor* papa, bool deep = false) const -> caf::result<sp_limpl> override;

	auto spawn_actor(sp_limpl limpl) const -> caf::actor override;

	auto data() -> obj_or_err override;
	auto data(unsafe_t) const -> sp_obj override;
	auto set_data(sp_obj obj) -> void;

	ENGINE_TYPE_DECL
};

struct BS_HIDDEN_API weak_link_impl : ilink_impl {
	std::weak_ptr<objbase> data_;

	using super = ilink_impl;

	weak_link_impl();
	weak_link_impl(std::string name, const sp_obj& data, Flags f);

	auto clone(link_actor* papa, bool deep = false) const -> caf::result<sp_limpl> override;

	auto spawn_actor(sp_limpl limpl) const -> caf::actor override;

	auto data() -> obj_or_err override;
	auto data(unsafe_t) const -> sp_obj override;
	auto set_data(const sp_obj& obj) -> void;

	auto propagate_handle() -> node_or_err override;

	ENGINE_TYPE_DECL
};

/*-----------------------------------------------------------------------------
 * fast link actor suitable for both hard & weak links
 * 1) contains direct ptr to object (data)
 * 2) access to object's data & node is always fast, s.t. we don't need to manage req status
 * 3) joins object's home group & react on transactions
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API hard_link_actor : public cached_link_actor {
public:
	using super = cached_link_actor;
	using super::super;

	// actor will join object's home
	using actor_type = super::actor_type::extend<
		// data altered ack from object
		caf::reacts_to<a_ack, a_data, tr_result::box>
	>;

	using typed_behavior = actor_type::behavior_type;

	// part of behavior overloaded by this actor
	using typed_behavior_overload = caf::typed_behavior<
		// get data
		caf::replies_to<a_data, bool>::with<result_or_errbox<sp_obj>>,
		// get data node
		caf::replies_to<a_data_node, bool>::with<node_or_errbox>,
		// request status change ack
		caf::reacts_to<a_ack, a_lnk_status, Req, ReqStatus, ReqStatus>,
		// data altered ack from object
		caf::reacts_to<a_ack, a_data, tr_result::box>
	>;

	// if object is already initialized, auto-join it's group
	hard_link_actor(caf::actor_config& cfg, caf::group self_grp, sp_limpl Limpl);

	auto make_typed_behavior() -> typed_behavior;
	auto make_behavior() -> behavior_type override;

private:
	std::string obj_hid_;
	caf::actor_addr obj_actor_addr_;

	auto monitor_object() -> void;
};

NAMESPACE_END(blue_sky::tree)
