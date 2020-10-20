/// @date 05.10.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/map_link.h>

#include "link_impl.h"
#include "link_actor.h"

#include <unordered_map>

NAMESPACE_BEGIN(blue_sky::tree)
class map_link_actor;

/*-----------------------------------------------------------------------------
 *  map_link impl
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API map_link_impl : public link_impl {
public:
	using super = link_impl;
	using sp_map_limpl = std::shared_ptr<map_link_impl>;
	using link_mapper_f = map_link::link_mapper_f;
	using link_or_node = map_link::link_or_node;

	// map link specific behavior
	using map_actor_type = caf::typed_actor<
		// delay output node refresh till next DataNode request
		caf::reacts_to<a_lazy, a_node_clear>,
		// immediately refresh output node & return it
		caf::replies_to<a_node_clear>::with<node_or_errbox>,
		// invoke mapper on given link from given origin node (sent by retranslator)
		caf::reacts_to<a_ack, a_apply, link /* src */>,
		// link erased from input (sub)node
		caf::reacts_to<a_ack, a_node_erase, lid_type /* ID of erased link */>
	>;
	// complete behavior
	using actor_type = super::actor_type::extend_with<map_actor_type>;

	///////////////////////////////////////////////////////////////////////////////
	//  base link_impl API
	//
	map_link_impl();
	map_link_impl(
		std::string name, link_mapper_f mf, link_or_node input, link_or_node output,
		Event update_on, TreeOpts opts, Flags f
	);

	auto spawn_actor(sp_limpl) const -> caf::actor override;

	auto clone(bool deep) const -> sp_limpl override;

	auto propagate_handle() -> node_or_err override;

	// return error/nullptr
	auto data() -> obj_or_err override;
	auto data(unsafe_t) const -> sp_obj override;

	///////////////////////////////////////////////////////////////////////////////
	//  additional map-specific API
	//
	// update/erase single link
	auto update(map_link_actor* self, link src_link) -> void;
	auto erase(map_link_actor* self, lid_type src_lid) -> void;
	// reset all mappings from scratch, started in separate `worker` actor
	auto refresh(map_link_actor* self, caf::event_based_actor* rworker) -> caf::result<error::box>;

	// data members
	link_mapper_f mf_;
	node in_, out_;
	// mapping from input link ID -> output link ID
	using io_map_t = std::unordered_map<lid_type, lid_type>;
	io_map_t io_map_;

	Event update_on_;
	TreeOpts opts_;

	ENGINE_TYPE_DECL
};

/*-----------------------------------------------------------------------------
 *  map_link_actor
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API map_link_actor : public link_actor {
public:
	using super = link_actor;

	using typed_map_behavior = map_link_impl::map_actor_type::behavior_type;
	using typed_behavior = map_link_impl::actor_type::behavior_type;

	using typed_behavior_overload = map_link_impl::map_actor_type::extend<
		caf::replies_to<a_data_node, bool>::with<node_or_errbox>
	>::behavior_type;

	using refresh_behavior_overload = caf::typed_behavior<
		caf::replies_to<a_data_node, bool>::with<node_or_errbox>
	>;

	map_link_actor(caf::actor_config& cfg, caf::group self_grp, sp_limpl Limpl);

	decltype(auto) mimpl() const { return static_cast<map_link_impl&>(impl); }

	auto name() const -> const char* override;

	auto make_casual_behavior() -> typed_behavior;
	auto make_refresh_behavior() -> typed_behavior;

	// returns refresh behavior
	auto make_behavior() -> behavior_type override;

private:
	caf::actor inp_listener_;

	auto reset_input_listener(Event update_on, TreeOpts opts) -> void;
};

NAMESPACE_END(blue_sky::tree)
