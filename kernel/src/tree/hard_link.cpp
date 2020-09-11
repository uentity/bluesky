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

#include "hard_link.h"

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;

/*-----------------------------------------------------------------------------
 *  hard_link_actor
 *-----------------------------------------------------------------------------*/
hard_link_actor::hard_link_actor(caf::actor_config& cfg, caf::group self_grp, sp_limpl Limpl) :
	super(cfg, std::move(self_grp), std::move(Limpl))
{
	// if object is already initialized, auto-join it's group
	if(auto obj = impl.data(unsafe)) {
		join(obj->home());
		obj_hid_ = obj->home_id();
	}
}

auto hard_link_actor::make_typed_behavior() -> typed_behavior {
	return first_then_second(typed_behavior_overload{
		// in hard link we can assume that Data & DataNode requests costs are very small (~0)
		// => override slow API to exclude extra possible delays
		[=](a_data, bool) -> obj_or_errbox {
			adbg(this) << "<- a_data fast, status = " <<
				to_string(impl.req_status(Req::Data)) << "," << to_string(impl.req_status(Req::DataNode)) << std::endl;

			return pimpl_->data().and_then([](auto&& obj) {
				return obj ?
					obj_or_errbox(std::move(obj)) :
					unexpected_err_quiet(Error::EmptyData);
			});
		},

		[=](a_data_node, bool) -> node_or_errbox {
			adbg(this) << "<- a_data_node fast, status = " <<
				to_string(impl.req_status(Req::Data)) << "," << to_string(impl.req_status(Req::DataNode)) << std::endl;

			return pimpl_->data().and_then([](const auto& obj) -> node_or_errbox {
				if(obj) {
					if(auto n = obj->data_node())
						return n;
					return unexpected_err_quiet(Error::NotANode);
				}
				return unexpected_err_quiet(Error::EmptyData);
			});
		},

		[=](a_ack, a_lnk_status, Req req, ReqStatus new_rs, ReqStatus prev_rs) {
			// join object's home group
			if(req == Req::Data && new_rs == ReqStatus::OK && obj_hid_.empty()) {
				if(auto obj = impl.data(unsafe)) {
					join(obj->home());
					obj_hid_ = obj->home_id();
				}
			}
			// retranslate ack to upper level
			ack_up(a_lnk_status(), req, new_rs, prev_rs);
		},

		[=](a_home, std::string new_hid) {
			// object's home ID changed
			if(!obj_hid_.empty())
				leave(system().groups().get_local(obj_hid_));

			join(system().groups().get_local(obj_hid_));
			obj_hid_ = std::move(new_hid);
		}
	}, super::make_typed_behavior());
}

auto hard_link_actor::make_behavior() -> behavior_type {
	return make_typed_behavior().unbox();
}

/*-----------------------------------------------------------------------------
 *  hard_link
 *-----------------------------------------------------------------------------*/
///////////////////////////////////////////////////////////////////////////////
//  impl
//
hard_link_impl::hard_link_impl(std::string name, sp_obj data, Flags f)
	: super(std::move(name), data, f)
{
	set_data(std::move(data));
}

hard_link_impl::hard_link_impl()
	: super()
{}

auto hard_link_impl::data() -> obj_or_err { return data_; }

auto hard_link_impl::data(unsafe_t) const -> sp_obj { return data_; }

auto hard_link_impl::set_data(sp_obj obj) -> void {
	if(data_ = std::move(obj); data_) {
		rs_reset(Req::Data, ReqReset::Always, ReqStatus::OK);
		if(data_->data_node())
			rs_reset(Req::DataNode, ReqReset::Always, ReqStatus::OK);
	}
	inode_ = make_inode(data_, inode_);
}

auto hard_link_impl::clone(bool deep) const -> sp_limpl {
	return std::make_shared<hard_link_impl>(
		name_, deep ? kernel::tfactory::clone_object(data_) : data_, flags_
	);
}

auto hard_link_impl::spawn_actor(sp_limpl limpl) const -> caf::actor {
	return spawn_lactor<hard_link_actor>(std::move(limpl));
}

///////////////////////////////////////////////////////////////////////////////
//  class
//
hard_link::hard_link(std::string name, sp_obj data, Flags f) :
	super(std::make_shared<hard_link_impl>(
		std::move(name), std::move(data), f
	))
{}

hard_link::hard_link(std::string name, node folder, Flags f) :
	super(std::make_shared<hard_link_impl>(
		std::move(name), std::make_shared<objnode>(std::move(folder)), f
	))
{}

hard_link::hard_link() :
	super(std::make_shared<hard_link_impl>(), false)
{}

LINK_CONVERT_TO(hard_link)
LINK_TYPE_DEF(hard_link, hard_link_impl, "hard_link")

/*-----------------------------------------------------------------------------
 *  weak_link
 *-----------------------------------------------------------------------------*/
///////////////////////////////////////////////////////////////////////////////
//  impl + actor
//
weak_link_impl::weak_link_impl(std::string name, const sp_obj& obj, Flags f)
	: super(std::move(name), obj, f)
{
	set_data(obj);
}

weak_link_impl::weak_link_impl()
	: super()
{}

auto weak_link_impl::data() -> obj_or_err {
	if(data_.expired())
		return unexpected_err_quiet(Error::LinkExpired);
	else if(auto obj = data_.lock())
		return obj;
	return unexpected_err_quiet(Error::EmptyData);
}

auto weak_link_impl::data(unsafe_t) const -> sp_obj { return data_.lock(); }

auto weak_link_impl::set_data(const sp_obj& obj) -> void {
	if(data_ = obj; obj) {
		// set status silently
		rs_reset(Req::Data, ReqReset::Always, ReqStatus::OK);
		if(obj->data_node())
			rs_reset(Req::DataNode, ReqReset::Always, ReqStatus::OK);
	}
	inode_ = make_inode(obj, inode_);
}

auto weak_link_impl::spawn_actor(sp_limpl limpl) const -> caf::actor {
	return spawn_lactor<hard_link_actor>(std::move(limpl));
}

auto weak_link_impl::clone(bool deep) const -> sp_limpl {
	// cannot make deep copy of object pointee
	return std::make_shared<weak_link_impl>(name_, data_.lock(), flags_);
}

auto weak_link_impl::propagate_handle() -> node_or_err {
	// weak link cannot be a node's handle
	return data().and_then([](const sp_obj& obj) -> node_or_err {
		if(auto N = obj->data_node())
			return N;
		return unexpected_err_quiet(Error::NotANode);
	});
	//return actorf<node_or_errbox>(super, a_data_node(), true);
}

///////////////////////////////////////////////////////////////////////////////
//  class
//
weak_link::weak_link(std::string name, const sp_obj& data, Flags f) :
	super(std::make_shared<weak_link_impl>(std::move(name), data, f))
{}

weak_link::weak_link() :
	super(std::make_shared<weak_link_impl>(), false)
{}

LINK_CONVERT_TO(weak_link)
LINK_TYPE_DEF(weak_link, weak_link_impl, "weak_link")

NAMESPACE_END(blue_sky::tree)
