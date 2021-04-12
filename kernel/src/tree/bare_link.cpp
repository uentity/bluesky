/// @date 21.09.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/bare_link.h>

#include "link_impl.h"
#include "nil_engine.h"

#include <optional>

NAMESPACE_BEGIN(blue_sky::tree)

bare_link::bare_link(sp_limpl impl) : pimpl_(std::move(impl)) {}

bare_link::bare_link(const link& rhs) : bare_link(rhs.bare()) {}

auto bare_link::operator =(const link& rhs) -> bare_link& {
	return (*this = rhs.bare());
}

auto bare_link::hash() const noexcept -> std::size_t {
	return std::hash<sp_limpl>{}(pimpl_);
}

auto bare_link::type_id() const -> std::string_view {
	return pimpl_->type_id();
}

auto bare_link::is_nil() const -> bool {
	return pimpl_ == nil_link::pimpl();
}

auto bare_link::armed() const -> link {
	return pimpl_->super_engine();
}

auto bare_link::id() const -> lid_type {
	return pimpl_->id_;
}

auto bare_link::owner() const -> node {
	return pimpl_->owner();
}

auto bare_link::info() -> result_or_err<inode> {
	return pimpl_->get_inode()
	.and_then([](const inodeptr& i) {
		return i ?
			result_or_err<inode>(*i) :
			tl::make_unexpected(error::quiet(Error::EmptyInode));
	});
}

auto bare_link::flags() const -> Flags {
	return pimpl_->flags_;
}

auto bare_link::req_status(Req request) const -> ReqStatus {
	return pimpl_->req_status(request);
}

auto bare_link::name() const -> std::string {
	return pimpl_->name_;
}

auto bare_link::oid() const -> std::string {
	if(auto obj = pimpl_->data(unsafe))
		return obj->id();
	return nil_oid;
}

auto bare_link::obj_type_id() const -> std::string {
	if(auto obj = pimpl_->data(unsafe))
		return obj->type_id();
	return nil_otid;
}

auto bare_link::data() -> sp_obj {
	return pimpl_->data(unsafe);
}

auto bare_link::data_node() -> node {
	return pimpl_->data_node(unsafe);
}

auto bare_link::data_node_if_ok() -> node {
	std::optional<node> res;
	pimpl_->rs_apply(
		Req::DataNode, [&](auto&) { res = pimpl_->data_node(unsafe); },
		ReqReset::IfEq, ReqStatus::OK
	);
	return res ? *res : node::nil();
}

auto bare_link::data_node_hid() -> std::string {
	if(auto N = data_node())
		return std::string{N.home_id()};
	else return {};
}

NAMESPACE_END(blue_sky::tree)
