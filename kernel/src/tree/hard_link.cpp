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

NAMESPACE_BEGIN(blue_sky::tree)

/*-----------------------------------------------------------------------------
 *  hard_link
 *-----------------------------------------------------------------------------*/
hard_link::hard_link(std::string name, sp_obj data, Flags f) :
	ilink(std::move(name), data, f), data_(std::move(data))
{
	if(data_) {
		rs_reset(Req::Data, ReqStatus::OK);
		if(data_->is_node())
			rs_reset(Req::DataNode, ReqStatus::OK);
	}
}

link::sp_link hard_link::clone(bool deep) const {
	return std::make_shared<hard_link>(
		name(),
		deep ? kernel::tfactory::clone_object(data_) : data_,
		flags()
	);
}

std::string hard_link::type_id() const {
	return "hard_link";
}

result_or_err<sp_obj> hard_link::data_impl() const {
	return data_;
}

/*-----------------------------------------------------------------------------
 *  weak_link
 *-----------------------------------------------------------------------------*/
weak_link::weak_link(std::string name, const sp_obj& data, Flags f) :
	ilink(std::move(name), data, f), data_(data)
{
	// can call `data_impl()` directly, because parallel requests aren't possible in ctor
	data_impl().map([this](const sp_obj& obj) {
		if(obj) {
			rs_reset(Req::Data, ReqStatus::OK);
			if(obj->is_node())
				rs_reset(Req::DataNode, ReqStatus::OK);
		}
	});
}

link::sp_link weak_link::clone(bool deep) const {
	// cannot make deep copy of object pointee
	return deep ? nullptr : std::make_shared<weak_link>(name(), data_.lock(), flags());
}

std::string weak_link::type_id() const {
	return "weak_link";
}

result_or_err<sp_obj> weak_link::data_impl() const {
	return data_.expired() ?
		tl::make_unexpected(error::quiet(Error::LinkExpired)) :
		result_or_err<sp_obj>(data_.lock());
}

auto weak_link::propagate_handle() -> result_or_err<sp_node> {
	// weak link cannot be a node's handle
	return data_node_ex();
}

NAMESPACE_END(blue_sky::tree)
