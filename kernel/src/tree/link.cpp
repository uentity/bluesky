/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief Implementation os link
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "link_impl.h"

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

/*-----------------------------------------------------------------------------
 *  inode
 *-----------------------------------------------------------------------------*/
inode::inode()
	: suid(false), sgid(false), sticky(false),
	u(7), g(5), o(5),
	mod_time(std::chrono::system_clock::now())
{}

/*-----------------------------------------------------------------------------
 *  link
 *-----------------------------------------------------------------------------*/
link::link(std::string name, Flags f)
	: pimpl_(std::make_unique<impl>(std::move(name), f))
{}

link::~link() {}

/// access link's unique ID
auto link::id() const -> const id_type& {
	return pimpl_->id_;
}

/// obtain link's symbolic name
auto link::name() const -> std::string {
	return pimpl_->name_;
}

/// get link's container
auto link::owner() const -> sp_node {
	return pimpl_->owner_.lock();
}

// get link's object ID
std::string link::oid() const {
	if(auto obj = data()) return obj->id();
	return boost::uuids::to_string(boost::uuids::nil_uuid());
}

std::string link::obj_type_id() const {
	if(auto obj = data()) return obj->type_id();
	return type_descriptor::nil().name;
}

void link::reset_owner(const sp_node& new_owner) {
	pimpl_->owner_ = new_owner;
}

link::Flags link::flags() const {
	return pimpl_->flags_;
}

void link::set_flags(Flags new_flags) {
	pimpl_->flags_ = new_flags;
}

auto link::rename(std::string new_name) -> void {
	pimpl_->rename(std::move(new_name));
}

auto link::rename_silent(std::string new_name) -> void {
	pimpl_->name_ = std::move(new_name);
}

const inode& link::info() const {
	return pimpl_->inode_;
}
inode& link::info() {
	return pimpl_->inode_;
}

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

