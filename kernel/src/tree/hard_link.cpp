/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief hard link implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/link.h>
#include <bs/tree/node.h>
#include <bs/tree/errors.h>
#include <bs/kernel/types_factory.h>

#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include "link_actor.h"

NAMESPACE_BEGIN(blue_sky::tree)

/*-----------------------------------------------------------------------------
 *  hard_link
 *-----------------------------------------------------------------------------*/
///////////////////////////////////////////////////////////////////////////////
//  actor
//
hard_link_actor::hard_link_actor(caf::actor_config& cfg, std::string name, sp_obj data, Flags f)
	: super(cfg, std::move(name), data, f), data_(std::move(data))
{
	if(data_) {
		rs_reset(Req::Data, ReqStatus::OK);
		rs_reset(Req::DataNode, data_->is_node() ? ReqStatus::OK : ReqStatus::Error);
	}
}

auto hard_link_actor::data() -> result_or_err<sp_obj> { return data_; }

// install simple behavior
auto hard_link_actor::make_behavior() -> behavior_type {
	return make_simple_behavior();
}

///////////////////////////////////////////////////////////////////////////////
//  class
//
hard_link::hard_link(std::string name, sp_obj data, Flags f) :
	ilink(spawn_lactor<hard_link_actor>(std::move(name), data, f))
{}

auto hard_link::pimpl() const -> hard_link_actor* {
	return static_cast<hard_link_actor*>(ilink::pimpl());
}

link::sp_link hard_link::clone(bool deep) const {
	return std::make_shared<hard_link>(
		name(),
		deep ? kernel::tfactory::clone_object(pimpl()->data_) : pimpl()->data_,
		flags()
	);
}

std::string hard_link::type_id() const {
	return "hard_link";
}

/*-----------------------------------------------------------------------------
 *  weak_link
 *-----------------------------------------------------------------------------*/
///////////////////////////////////////////////////////////////////////////////
//  actor
//
weak_link_actor::weak_link_actor(caf::actor_config& cfg, std::string name, const sp_obj& obj, Flags f)
	: super(cfg, std::move(name), obj, f), data_(obj)
{
	data().map([this](const sp_obj& obj) {
		if(obj) {
			rs_reset(Req::Data, ReqStatus::OK);
			rs_reset(Req::DataNode, obj->is_node() ? ReqStatus::OK : ReqStatus::Error);
		}
	});
}

auto weak_link_actor::data() -> result_or_err<sp_obj> {
	using result_t = result_or_err<sp_obj>;
	return data_.expired() ?
		tl::make_unexpected(error::quiet(Error::LinkExpired)) :
		result_t{ data_.lock() };
}

// install simple behavior
auto weak_link_actor::make_behavior() -> behavior_type {
	return make_simple_behavior();
}

///////////////////////////////////////////////////////////////////////////////
//  class
//
weak_link::weak_link(std::string name, const sp_obj& data, Flags f) :
	ilink(spawn_lactor<weak_link_actor>(std::move(name), data, f))
{}

auto weak_link::pimpl() const -> weak_link_actor* {
	return static_cast<weak_link_actor*>(ilink::pimpl());
}
link::sp_link weak_link::clone(bool deep) const {
	// cannot make deep copy of object pointee
	return deep ? nullptr : std::make_shared<weak_link>(name(), pimpl()->data_.lock(), flags());
}

std::string weak_link::type_id() const {
	return "weak_link";
}

auto weak_link::propagate_handle() -> result_or_err<sp_node> {
	// weak link cannot be a node's handle
	return data_node_ex();
}

NAMESPACE_END(blue_sky::tree)
