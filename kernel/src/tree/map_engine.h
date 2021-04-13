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
///////////////////////////////////////////////////////////////////////////////
//  base
//
class BS_HIDDEN_API map_impl_base : public link_impl {
public:
	using super = link_impl;
	using sp_map_impl_base = std::shared_ptr<map_impl_base>;

	// map link specific behavior
	using map_actor_type = caf::typed_actor<
		// tells that outpt node has actual cache & we must switch to normal behavior
		caf::replies_to<a_mlnk_fresh>::with<bool>,
		// delay output node refresh till next DataNode request
		caf::reacts_to<a_lazy, a_node_clear>,
		// immediately refresh output node & return it
		caf::replies_to<a_node_clear>::with<node_or_errbox>,
		// invoke mapper on given link from given origin node (sent by retranslator)
		caf::reacts_to<a_ack, a_apply, lid_type /* src */, event>,
		// link erased from input (sub)node
		caf::reacts_to<a_ack, a_node_erase, lid_type /* ID of erased link */, event>
	>;
	// complete behavior
	using actor_type = super::actor_type::extend_with<map_actor_type>;

	///////////////////////////////////////////////////////////////////////////////
	//  base link_impl API
	//
	map_impl_base(bool is_link_mapper);

	map_impl_base(
		bool is_link_mapper, uuid tag, std::string name,
		const link_or_node& input, const link_or_node& output,
		Event update_on, TreeOpts opts, Flags f
	);

	auto spawn_actor(sp_limpl) const -> caf::actor override;

	auto propagate_handle() -> node_or_err override;

	// return error/nullptr
	auto data() -> obj_or_err override;
	auto data(unsafe_t) const -> sp_obj override;
	// returns output directory
	auto data_node(unsafe_t) const -> node override;

	///////////////////////////////////////////////////////////////////////////////
	//  map-specific API
	//
	// update/erase single link
	virtual auto update(map_link_actor* papa, link src_link, event ev) -> void = 0;
	virtual auto erase(map_link_actor* papa, lid_type src_lid, event ev) -> void = 0;
	// reset all mappings from scratch, started in separate `worker` actor
	virtual auto refresh(map_link_actor* papa) -> caf::result<node_or_errbox> = 0;

	// data members
	node in_, out_;
	uuid tag_;
	Event update_on_;
	TreeOpts opts_;
	const bool is_link_mapper;
};

///////////////////////////////////////////////////////////////////////////////
//  map link -> link
//
class BS_HIDDEN_API map_link_impl : public map_impl_base {
public:
	using link_mapper_f = map_link::link_mapper_f;
	using sp_map_link_impl = std::shared_ptr<map_link_impl>;

	map_link_impl();

	template<typename... Args>
	explicit map_link_impl(link_mapper_f mf, Args&&... args) :
		map_impl_base(true, std::forward<Args>(args)...), mf_(std::move(mf))
	{}

	auto clone(link_actor* papa, bool deep) const -> caf::result<sp_limpl> override final;

	// update/erase single link
	auto update(map_link_actor* papa, link src_link, event ev) -> void override final;
	auto erase(map_link_actor* papa, lid_type src_lid, event ev) -> void override final;
	// reset all mappings from scratch, started in separate `worker` actor
	auto refresh(map_link_actor* papa) -> caf::result<node_or_errbox> override final;
	// refresh impl inside worker actor
	auto refresh(map_link_actor* papa, caf::event_based_actor* rworker)
	-> caf::result<node_or_errbox>;

	link_mapper_f mf_;
	// mapping from input link ID -> output link ID
	using io_map_t = std::unordered_map<lid_type, lid_type>;
	io_map_t io_map_;

	ENGINE_TYPE_DECL
};

///////////////////////////////////////////////////////////////////////////////
//  map node -> node
//
class BS_HIDDEN_API map_node_impl : public map_impl_base {
public:
	using node_mapper_f = map_link::node_mapper_f;
	using sp_map_node_impl = std::shared_ptr<map_link_impl>;

	map_node_impl();

	template<typename... Args>
	explicit map_node_impl(node_mapper_f mf, Args&&... args) :
		map_impl_base(false, std::forward<Args>(args)...), mf_(std::move(mf))
	{}

	auto clone(link_actor* papa, bool deep) const -> caf::result<sp_limpl> override final;

	// implementation is identical in all cases - just spawn mapper job
	auto update(map_link_actor* papa, link src_link, event ev) -> void override final;
	auto erase(map_link_actor* papa, lid_type src_lid, event ev) -> void override final;
	auto refresh(map_link_actor* papa) -> caf::result<node_or_errbox> override final;

	node_mapper_f mf_;

	ENGINE_TYPE_DECL
};

/*-----------------------------------------------------------------------------
 *  map_link_actor
 *-----------------------------------------------------------------------------*/
class BS_HIDDEN_API map_link_actor : public link_actor {
public:
	using super = link_actor;
	using io_map_t = map_link_impl::io_map_t;

	using typed_map_behavior = map_link_impl::map_actor_type::behavior_type;
	using typed_behavior = map_link_impl::actor_type::behavior_type;

	using typed_behavior_overload = map_link_impl::map_actor_type::extend<
		caf::replies_to<a_data, bool>::with<obj_or_errbox>,
		caf::replies_to<a_data_node, bool>::with<node_or_errbox>
	>::behavior_type;

	using refresh_behavior_overload = caf::typed_behavior<
		caf::replies_to<a_mlnk_fresh>::with<bool>,
		caf::replies_to<a_data_node, bool>::with<node_or_errbox>,
		// override update to trigger refresh first
		caf::reacts_to<a_ack, a_apply, lid_type /* src */, event>,
		caf::reacts_to<a_ack, a_node_erase, lid_type /* ID of erased link */, event>
	>;

	map_link_actor(caf::actor_config& cfg, caf::group self_grp, sp_limpl Limpl);

	decltype(auto) mimpl() const { return static_cast<map_impl_base&>(impl); }

	auto on_exit() -> void override;

	auto make_casual_behavior() -> typed_behavior;
	auto make_refresh_behavior() -> refresh_behavior_overload;

	// returns refresh behavior
	auto make_behavior() -> behavior_type override;

private:
	caf::actor inp_listener_;

	auto reset_input_listener(Event update_on, TreeOpts opts) -> void;
};

NAMESPACE_END(blue_sky::tree)
