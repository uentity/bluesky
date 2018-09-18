/// @file
/// @author uentity
/// @date 10.08.2018
/// @brief Fusion link implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

//#include <bs/kernel.h>
//#include <bs/tree/fusion.h>
//#include <bs/tree/node.h>

#include "link_impl.h"
#include "fusion_link_impl.h"

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

/*-----------------------------------------------------------------------------
 *  fusion_link::impl
 *-----------------------------------------------------------------------------*/
fusion_link::fusion_link(
	std::string name, sp_node data, sp_fusion bridge, Flags f
) :
	link(std::move(name), f),
	pimpl_(std::make_unique<impl>(std::move(bridge), std::move(data)))
{
	if(pimpl_->data_) {
		rs_reset(Req::Data, ReqStatus::OK);
		rs_reset(Req::DataNode, ReqStatus::OK);
	}
}

fusion_link::fusion_link(
	std::string name, const char* obj_type, std::string oid, sp_fusion bridge, Flags f
) :
	link(std::move(name), f),
	pimpl_(std::make_unique<impl>(
		std::move(bridge), BS_KERNEL.create_object(obj_type, std::move(oid))
	))
{
	if(pimpl_->data_) {
		rs_reset(Req::Data, ReqStatus::OK);
		rs_reset(Req::DataNode, ReqStatus::OK);
	}
	else
		bserr() << log::E("fusion_link: cannot create object of type '{}'! Empty link!") << obj_type << log::end;
}

fusion_link::~fusion_link() {}

auto fusion_link::clone(bool deep) const -> sp_link {
	auto res = std::make_shared<fusion_link>(
		name(),
		deep ? BS_KERNEL.clone_object(std::static_pointer_cast<objbase>(pimpl_->data_)) : pimpl_->data_,
		pimpl_->bridge_, flags()
	);
	return res;
}

auto fusion_link::type_id() const -> std::string {
	return "fusion_link";
}

auto fusion_link::data_impl() const -> result_or_err<sp_obj> {
	if(req_status(Req::Data) == ReqStatus::OK) {
		return pimpl_->data_;
	}
	if(const auto B = bridge()) {
		auto err = B->pull_data(pimpl_->data_);
		return err.ok() ? result_or_err<sp_obj>(pimpl_->data_) : tl::make_unexpected(std::move(err));
	}
	return tl::make_unexpected("Fusion bridge isn't set");
}

auto fusion_link::data_node_impl() const -> result_or_err<sp_node> {
	if(req_status(Req::DataNode) == ReqStatus::OK) {
		return pimpl_->data_;
	}
	if(const auto B = bridge()) {
		auto err = B->populate(pimpl_->data_);
		return err.ok() ? result_or_err<sp_node>(pimpl_->data_) : tl::make_unexpected(std::move(err));
	}
	return tl::make_unexpected("Fusion bridge isn't set");
}

auto fusion_link::populate(const std::string& child_type_id) -> error {
	// start populating only if link isn't already being populated
	if(rs_reset_if_neq(Req::DataNode, ReqStatus::Busy, ReqStatus::Busy) != ReqStatus::Busy) {
		const auto B = bridge();
		if(!B) return {"Fusion bridge isn't set"};
		const auto err = B->populate(pimpl_->data_, child_type_id);
		// populate raises structure populated status
		if(err.ok()) {
			pimpl_->data_ ?
				rs_reset(Req::DataNode, ReqStatus::OK) :
				rs_reset(Req::DataNode, ReqStatus::Void);
		}
		else rs_reset(Req::DataNode, ReqStatus::Error);
		return err;
	}
	return error::quiet("Link is busy");
}

auto fusion_link::bridge() const -> sp_fusion {
	if(pimpl_->bridge_) return pimpl_->bridge_;
	// try to look up in parent link
	if(auto parent = owner()) {
		if(auto phandle = parent->handle()) {
			if(phandle->type_id() == "fusion_link") {
				return std::static_pointer_cast<fusion_link>(phandle)->bridge();
			}
		}
	}
	return nullptr;
}

auto fusion_link::reset_bridge(sp_fusion new_bridge) -> void {
	pimpl_->reset_bridge(std::move(new_bridge));
}

NAMESPACE_END(blue_sky) NAMESPACE_END(tree)

