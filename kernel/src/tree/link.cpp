/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief Implementation os link
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel/config.h>
#include "link_impl.h"

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  misc
 *-----------------------------------------------------------------------------*/
link::link(std::string name, Flags f)
	: pimpl_(std::make_unique<impl>(std::move(name), f))
{}

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

void link::reset_owner(const sp_node& new_owner) {
	std::lock_guard<std::mutex> g(pimpl_->solo_);
	pimpl_->owner_ = new_owner;
}

auto link::info() const -> result_or_err<inode> {
	return get_inode().and_then([](const inodeptr& i) {
		return i ?
			result_or_err<inode>(*i) :
			tl::make_unexpected(error::quiet(Error::EmptyInode));
	});
}

auto link::get_inode() const -> result_or_err<inodeptr> {
	// default implementation obtains inode from `data_ex()->inode_`
	return data_ex().and_then([](const sp_obj& obj) {
		return obj ?
			result_or_err<inodeptr>(obj->inode_.lock()) :
			tl::make_unexpected(error::quiet(Error::EmptyData));
	});
}

auto link::make_inode(const sp_obj& obj, inodeptr new_i) -> inodeptr {
	if(!obj) return nullptr;

	auto obj_i = obj->inode_.lock();
	if(!obj_i) {
		obj_i = new_i ? std::move(new_i) : std::make_shared<inode>();
		obj->inode_ = obj_i;
	}
	else if(new_i) {
		*obj_i = *new_i;
	}
	return obj_i;
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
	).and_then([](sp_obj&& obj) {
		return obj ?
			result_or_err<sp_obj>(std::move(obj)) :
			tl::make_unexpected(error::quiet(Error::EmptyData));
	});
}

result_or_err<sp_node> link::data_node_ex(bool wait_if_busy) const {
	// never returns NULL node
	return link_invoke(
		this,
		[](const link* lnk) { return lnk->data_node_impl(); },
		pimpl_->status_[1], wait_if_busy
	).and_then([](sp_node&& N) {
		return N ?
			result_or_err<sp_node>(std::move(N)) :
			tl::make_unexpected(error::quiet(Error::NotANode));
	});
}

void link::self_handle_node(const sp_node& N) {
	if(N) N->set_handle(shared_from_this());
}

result_or_err<sp_node> link::propagate_handle() {
	return data_node_ex().and_then([this](sp_node&& N) -> result_or_err<sp_node> {
		N->set_handle(shared_from_this());
		return std::move(N);
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

auto link::data(process_data_cb f, bool high_priority) const -> void {
	high_priority ?
		pimpl_->send<caf::message_priority::high>(
			lnk_data_atom(), shared_from_this(), std::move(f)
		) :
		pimpl_->send(lnk_data_atom(), shared_from_this(), std::move(f));
}

auto link::data_node(process_data_cb f, bool high_priority) const -> void {
	high_priority ?
		pimpl_->send<caf::message_priority::high>(
			lnk_dnode_atom(), shared_from_this(), std::move(f)
		) :
		pimpl_->send(lnk_dnode_atom(), shared_from_this(), std::move(f));
}

/*-----------------------------------------------------------------------------
 *  ilink
 *-----------------------------------------------------------------------------*/
ilink::ilink(std::string name, const sp_obj& data, Flags f)
	: link(std::move(name), f), inode_(make_inode(data))
{}

auto ilink::get_inode() const -> result_or_err<inodeptr> {
	return inode_;
}

NAMESPACE_END(blue_sky::tree)
