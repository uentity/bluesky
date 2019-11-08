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

OMIT_OBJ_SERIALIZATION

NAMESPACE_BEGIN(blue_sky::tree)
using bs_detail::shared;

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

auto hard_link_impl::data() -> result_or_err<sp_obj> { return data_; }

auto hard_link_impl::set_data(sp_obj obj) -> void {
	auto guard = lock();

	inode_ = make_inode(obj, inode_);
	if(data_ = std::move(obj); data_) {
		// set status silently
		rs_reset(Req::Data, ReqReset::Always | ReqReset::Silent, ReqStatus::OK);
		rs_reset(
			Req::DataNode, ReqReset::Always | ReqReset::Silent,
			data_->is_node() ? ReqStatus::OK : ReqStatus::Error
		);
		//std::cout << "hard link " << to_string(id_) << ", name " << name_ << ": impl created " <<
		//	(int)status_[0].value << (int)status_[1].value << std::endl;
	}
}

auto hard_link_impl::spawn_actor(std::shared_ptr<link_impl> limpl) const -> caf::actor {
	return spawn_lactor<simple_link_actor>(std::move(limpl));
}

///////////////////////////////////////////////////////////////////////////////
//  class
//
hard_link::hard_link(std::string name, sp_obj data, Flags f) :
	super(std::make_shared<hard_link_impl>(std::move(name), data, f))
{}

hard_link::hard_link() :
	super(std::make_shared<hard_link_impl>(), false)
{}

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

auto hard_link::pimpl() const -> hard_link_impl* {
	return static_cast<hard_link_impl*>(super::pimpl());
}

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

auto weak_link_impl::data() -> result_or_err<sp_obj> {
	using result_t = result_or_err<sp_obj>;
	return data_.expired() ?
		tl::make_unexpected(error::quiet(Error::LinkExpired)) :
		result_t{ data_.lock() };
}

auto weak_link_impl::set_data(const sp_obj& obj) -> void {
	auto guard = lock();

	inode_ = make_inode(obj, inode_);
	if(data_ = obj; obj) {
		// set status silently
		rs_reset(Req::Data, ReqReset::Always | ReqReset::Silent, ReqStatus::OK);
		rs_reset(
			Req::DataNode, ReqReset::Always | ReqReset::Silent,
			obj->is_node() ? ReqStatus::OK : ReqStatus::Error
		);
	}
}

auto weak_link_impl::spawn_actor(std::shared_ptr<link_impl> limpl) const -> caf::actor {
	return spawn_lactor<simple_link_actor>(std::move(limpl));
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

link::sp_link weak_link::clone(bool deep) const {
	// cannot make deep copy of object pointee
	return deep ? nullptr : std::make_shared<weak_link>(name(), pimpl()->data_.lock(), flags());
}

std::string weak_link::type_id() const {
	return "weak_link";
}

auto weak_link::pimpl() const -> weak_link_impl* {
	return static_cast<weak_link_impl*>(super::pimpl());
}

auto weak_link::propagate_handle() -> result_or_err<sp_node> {
	// weak link cannot be a node's handle
	return data_node_ex();
}

NAMESPACE_END(blue_sky::tree)
