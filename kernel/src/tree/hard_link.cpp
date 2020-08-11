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

auto hard_link_impl::data(unsafe_t) -> sp_obj { return data_; }

auto hard_link_impl::set_data(sp_obj obj) -> void {
	if(data_ = std::move(obj); data_) {
		rs_reset(Req::Data, ReqReset::Always, ReqStatus::OK);
		if(data_->data_node())
			rs_reset(Req::DataNode, ReqReset::Always, ReqStatus::OK);
		//std::cout << "hard link " << to_string(id_) << ", name " << name_ << ": impl created " <<
		//	(int)status_[0].value << (int)status_[1].value << std::endl;
	}
	inode_ = make_inode(data_, inode_);
}

auto hard_link_impl::clone(bool deep) const -> sp_limpl {
	return std::make_shared<hard_link_impl>(
		name_, deep ? kernel::tfactory::clone_object(data_) : data_, flags_
	);
}

auto hard_link_impl::spawn_actor(sp_limpl limpl) const -> caf::actor {
	return spawn_lactor<fast_link_actor>(std::move(limpl));
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

auto weak_link_impl::data(unsafe_t) -> sp_obj { return data_.lock(); }

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
	return spawn_lactor<fast_link_actor>(std::move(limpl));
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
