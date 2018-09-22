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
{
	// run actor
	pimpl_->actor_ = BS_KERNEL.actor_system().spawn(impl::async_api);
	// connect actor with sender
	pimpl_->init_sender();
}

link::~link() {}

auto link::pimpl() const -> impl* {
	return pimpl_.get();
}

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

void link::reset_owner(sp_node new_owner) {
	std::lock_guard<std::mutex> g(pimpl_->solo_);
	pimpl_->owner_ = std::move(new_owner);
}

auto link::info() const -> inode {
	return pimpl_->inode_;
}
auto link::set_info(inode I) -> void {
	std::lock_guard<std::mutex> g(pimpl_->solo_);
	pimpl_->inode_ = std::move(I);
	// [TODO] send message that inode changed
}

link::Flags link::flags() const {
	return pimpl_->flags_;
}

void link::set_flags(Flags new_flags) {
	std::lock_guard<std::mutex> g(pimpl_->solo_);
	pimpl_->flags_ = new_flags;
}

auto link::rename(std::string new_name) -> void {
	pimpl_->rename(std::move(new_name));
}

auto link::rename_silent(std::string new_name) -> void {
	pimpl_->rename_silent(std::move(new_name));
}

/*-----------------------------------------------------------------------------
 *  sync API
 *-----------------------------------------------------------------------------*/
// get link's object ID
std::string link::oid() const {
	if(pimpl_->req_status(Req::Data) == ReqStatus::OK) {
		if(auto D = data()) return D->id();
	}
	return boost::uuids::to_string(boost::uuids::nil_uuid());
}

std::string link::obj_type_id() const {
	if(pimpl_->req_status(Req::Data) == ReqStatus::OK) {
		if(auto D = data()) return D->type_id();
	}
	return type_descriptor::nil().name;
}

result_or_err<sp_node> link::data_node_impl() const {
	return data_ex().and_then([](const sp_obj& obj){
		return obj ?

			obj->is_node() ?
				result_or_err<sp_node>(std::static_pointer_cast<tree::node>(obj)) :
				tl::make_unexpected(error::quiet(Error::NotANode)) :

			tl::make_unexpected(error::quiet(Error::EmptyData));
	});
}

result_or_err<sp_obj> link::data_ex(bool wait_if_busy) const {
	// never returns NULL object
	return link_invoke(
		this,
		[](const link* lnk) { return lnk->data_impl(); },
		pimpl_->status_[0], wait_if_busy
	).and_then([](const sp_obj&& obj) {
		return obj ?
			result_or_err<sp_obj>(std::move(obj)) : tl::make_unexpected(error::quiet(Error::EmptyData));
	});
}

result_or_err<sp_node> link::data_node_ex(bool wait_if_busy) const {
	// never returns NULL node
	return link_invoke(
		this,
		[](const link* lnk) { return lnk->data_node_impl(); },
		pimpl_->status_[1], wait_if_busy
	).and_then([](const sp_node&& N) {
		return N ?
			result_or_err<sp_node>(std::move(N)) : tl::make_unexpected(error::quiet(Error::NotANode));
	});
}

/*-----------------------------------------------------------------------------
 *  async API
 *-----------------------------------------------------------------------------*/
auto link::req_status(Req request) const -> ReqStatus {
	return pimpl_->req_status(request);
}

auto link::rs_reset(Req request, ReqStatus new_rs) const -> ReqStatus {
	return pimpl_->rs_reset(request, new_rs);
}

auto link::rs_reset_if_eq(Req request, ReqStatus self, ReqStatus new_rs) const -> ReqStatus {
	return pimpl_->rs_reset_if_eq(request, self, new_rs);
}

auto link::rs_reset_if_neq(Req request, ReqStatus self, ReqStatus new_rs) const -> ReqStatus {
	return pimpl_->rs_reset_if_neq(request, self, new_rs);
}

auto link::data(process_data_cb f, bool wait_if_busy) const -> void {
	pimpl_->send(lnk_data_atom(), this->shared_from_this(), std::move(f), wait_if_busy);
}

auto link::data_node(process_data_cb f, bool wait_if_busy) const -> void {
	pimpl_->send(lnk_dnode_atom(), this->shared_from_this(), std::move(f), wait_if_busy);
}

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

