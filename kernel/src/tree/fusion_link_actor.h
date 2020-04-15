/// @file
/// @author uentity
/// @date 15.08.2018
/// @brief Impl part of fusion_link PIMPL
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/fusion.h>
#include <bs/tree/node.h>
#include <bs/tree/errors.h>

#include "link_actor.h"

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::sp_fusion)

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  fusion_link_impl
 *-----------------------------------------------------------------------------*/
struct BS_HIDDEN_API fusion_link_impl : public ilink_impl {
	// treat Error::OKOK status as object is fully loaded by fusion_iface
	inline static const auto obj_fully_loaded = make_error_code(Error::OKOK);
	// bridge
	sp_fusion bridge_;
	// contained object
	sp_node data_;

	using actor_type = link_impl::actor_type::extend_with<fusion_link::fusion_actor_type>;

	using super = ilink_impl;
	using super::owner_;

	fusion_link_impl(std::string name, sp_node data, sp_fusion bridge, Flags f);
	fusion_link_impl();

	// search for valid (non-null) bridge up the tree
	auto bridge() const -> sp_fusion;

	auto reset_bridge(sp_fusion&& new_bridge) -> void;

	// implement `data`
	auto data() -> result_or_err<sp_obj> override;

	// unsafe version returns cached value
	auto data(unsafe_t) -> sp_obj override;

	// populate with specified child type
	auto populate(const std::string& child_type_id = "", bool wait_if_busy = true)
	-> result_or_err<sp_node>;

	auto spawn_actor(std::shared_ptr<link_impl> limpl) const -> caf::actor override;

	auto clone(bool deep = false) const -> sp_limpl override;

	auto propagate_handle(const link&) -> result_or_err<sp_node> override;

	LIMPL_TYPE_DECL
};

/*-----------------------------------------------------------------------------
 *  fusion_link_actor
 *-----------------------------------------------------------------------------*/
struct BS_HIDDEN_API fusion_link_actor : public cached_link_actor {
	using super = cached_link_actor;
	using super::super;

	using actor_type = fusion_link_impl::actor_type;
	using typed_behavior = actor_type::behavior_type;
	// part of behavior overloaded/added from super actor type
	using typed_behavior_overload = caf::typed_behavior<
		caf::replies_to<a_flnk_populate, std::string, bool>::with<result_or_errbox<sp_node>>,
		caf::replies_to<a_flnk_bridge>::with<sp_fusion>,
		caf::reacts_to<a_flnk_bridge, sp_fusion>,

		// get pointee OID
		caf::replies_to<a_lnk_oid>::with<std::string>,
		// get pointee type ID
		caf::replies_to<a_lnk_otid>::with<std::string>,
		// get pointee node group ID
		caf::replies_to<a_node_gid>::with<result_or_errbox<std::string>>
	>;

	auto fimpl() -> fusion_link_impl& { return static_cast<fusion_link_impl&>(impl); }

	// both Data & DataNode executes with `HasDataCache` flag set
	auto data_ex(obj_processor_f cb, ReqOpts opts) -> void override;

	// `data_node` just calls `populate`
	auto data_node_ex(node_processor_f cb, ReqOpts opts) -> void override;

	auto make_typed_behavior() -> typed_behavior;

	auto make_behavior() -> behavior_type override;
};

NAMESPACE_END(blue_sky::tree)
