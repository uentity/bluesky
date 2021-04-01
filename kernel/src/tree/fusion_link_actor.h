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
	// contained object
	sp_obj data_;

	using actor_type = link_impl::actor_type::extend_with<fusion_link::fusion_actor_type>;

	using super = ilink_impl;

	fusion_link_impl(std::string name, sp_obj data, sp_fusion bridge, Flags f);
	fusion_link_impl();

	auto spawn_actor(std::shared_ptr<link_impl> limpl) const -> caf::actor override;

	auto clone(link_actor* papa, bool deep = false) const -> caf::result<sp_limpl> override;

	// search for valid (non-null) bridge up the tree
	// [NOTE] protected by mutex
	auto bridge() const -> sp_fusion;

	auto reset_bridge(sp_fusion&& new_bridge) -> void;

	// link API
	auto data() -> obj_or_err override;
	auto data(unsafe_t) const -> sp_obj override;

	// fusion API
	auto pull_data(prop::propdict params) -> obj_or_err;
	auto populate(prop::propdict params) -> node_or_err;

	ENGINE_TYPE_DECL

private:
	friend atomizer;
	// bridge
	sp_fusion bridge_;
	mutable engine_impl_mutex bridge_guard_;
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
		// link API
		caf::replies_to<a_data, bool>::with<obj_or_errbox>,
		caf::replies_to<a_data_node, bool>::with<node_or_errbox>,
		// Fusion API
		caf::replies_to<a_flnk_data, prop::propdict, bool>::with<obj_or_errbox>,
		caf::replies_to<a_flnk_populate, prop::propdict, bool>::with<node_or_errbox>,
		// bridge get/set
		caf::replies_to<a_flnk_bridge>::with<sp_fusion>,
		caf::reacts_to<a_flnk_bridge, sp_fusion>
	>;

	auto fimpl() -> fusion_link_impl& { return static_cast<fusion_link_impl&>(impl); }

	auto make_typed_behavior() -> typed_behavior;
	auto make_behavior() -> behavior_type override;

private:
	auto make_ropts(Req r) -> ReqOpts;
};

NAMESPACE_END(blue_sky::tree)
