/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief hard link implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/link.h>
#include <bs/node.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

/*-----------------------------------------------------------------------------
 *  hard_link
 *-----------------------------------------------------------------------------*/
hard_link::hard_link(std::string name, sp_obj data) :
	link(std::move(name)), data_(std::move(data))
{}

link::sp_link hard_link::clone() const {
	return std::make_shared< hard_link >(*this);
}

sp_obj hard_link::data() const {
	return data_;
}

std::string hard_link::type_id() const {
	return "hard_link";
}

std::string hard_link::oid() const {
	return data_->id();
}

std::string hard_link::obj_type_id() const {
	return data_->type_id();
}

sp_node hard_link::data_node() const {
	return data_->is_node() ? std::static_pointer_cast<tree::node>(data_) : nullptr;
}

inode hard_link::info() const {
	return data_->info();
}

void hard_link::set_info(inodeptr i) {
	data_->set_info(std::move(i));
}

/*-----------------------------------------------------------------------------
 *  weak_link
 *-----------------------------------------------------------------------------*/
weak_link::weak_link(std::string name, const sp_obj& data) :
	link(std::move(name)), data_(data)
{}

link::sp_link weak_link::clone() const {
	return std::make_shared< weak_link >(*this);
}

sp_obj weak_link::data() const {
	return data_.lock();
}

std::string weak_link::type_id() const {
	return "weak_link";
}

std::string weak_link::oid() const {
	return data_.lock()->id();
}

std::string weak_link::obj_type_id() const {
	return data_.lock()->type_id();
}

sp_node weak_link::data_node() const {
	auto sdata = data_.lock();
	return sdata->is_node() ? std::static_pointer_cast<tree::node>(sdata) : nullptr;
}

inode weak_link::info() const {
	return data_.lock()->info();
}

void weak_link::set_info(inodeptr i) {
	data_.lock()->set_info(std::move(i));
}

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

