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
#include <bs/kernel.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

/*-----------------------------------------------------------------------------
 *  hard_link
 *-----------------------------------------------------------------------------*/
hard_link::hard_link(std::string name, sp_obj data, Flags f) :
	link(std::move(name), f), data_(std::move(data))
{}

link::sp_link hard_link::clone(bool deep) const {
	return std::make_shared<hard_link>(
		name_,
		deep ? BS_KERNEL.clone_object(data_) : data_,
		flags_
	);
}

std::string hard_link::type_id() const {
	return "hard_link";
}

result_or_err<sp_obj> hard_link::data_ex() const {
	return data_;
}

result_or_err<sp_node> hard_link::data_node_ex() const {
	return data_->is_node() ? std::static_pointer_cast<tree::node>(data_) : nullptr;
}

/*-----------------------------------------------------------------------------
 *  weak_link
 *-----------------------------------------------------------------------------*/
weak_link::weak_link(std::string name, const sp_obj& data, Flags f) :
	link(std::move(name), f), data_(data)
{}

link::sp_link weak_link::clone(bool deep) const {
	// cannot make deep copy of object pointee
	return deep ? nullptr : std::make_shared<weak_link>(name_, data_.lock(), flags_);
}

std::string weak_link::type_id() const {
	return "weak_link";
}

result_or_err<sp_obj> weak_link::data_ex() const {
	return data_.lock();
}

result_or_err<sp_node> weak_link::data_node_ex() const {
	auto sdata = data_.lock();
	return sdata->is_node() ? std::static_pointer_cast<tree::node>(sdata) : nullptr;
}

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

