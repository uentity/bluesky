/// @file
/// @author uentity
/// @date 15.04.2020
/// @brief Fusion link actor impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "fusion_link_actor.h"
#include "request_impl.h"

NAMESPACE_BEGIN(blue_sky::tree)
using namespace allow_enumops;
using bs_detail::shared;

/*-----------------------------------------------------------------------------
 *  fusion_link impl
 *-----------------------------------------------------------------------------*/
fusion_link_impl::fusion_link_impl(std::string name, sp_node data, sp_fusion bridge, Flags f) :
	super(std::move(name), data, f), bridge_(std::move(bridge)), data_(std::move(data))
{}

fusion_link_impl::fusion_link_impl() :
	super()
{}

// search for valid (non-null) bridge up the tree
auto fusion_link_impl::bridge() const -> sp_fusion {
	if(bridge_) return bridge_;
	// try to look up in parent link
	if(auto parent = owner_.lock()) {
		if(auto phandle = fusion_link{ parent->handle() })
			return phandle.bridge();
	}
	return nullptr;
}

auto fusion_link_impl::reset_bridge(sp_fusion&& new_bridge) -> void {
	bridge_ = std::move(new_bridge);
}

// implement `data`
auto fusion_link_impl::data() -> result_or_err<sp_obj> {
	if(req_status(Req::Data) == ReqStatus::OK)
		return data_;
	if(const auto B = bridge()) {
		if(!data_) data_ = std::make_shared<node>();
		auto err = B->pull_data(data_);
		if(err.code == obj_fully_loaded)
			rs_reset(Req::DataNode, ReqReset::IfNeq, ReqStatus::OK, ReqStatus::Busy);
		return err.ok() ? result_or_err<sp_obj>(data_) : tl::make_unexpected(std::move(err));
	}
	return tl::make_unexpected(error{Error::NoFusionBridge});
}

// unsafe version returns cached value
auto fusion_link_impl::data(unsafe_t) -> sp_obj {
	return data_;
}

// populate with specified child type
auto fusion_link_impl::populate(const std::string& child_type_id, bool wait_if_busy)
-> result_or_err<sp_node> {
	// assume that if `child_type_id` is nonepmty,
	// then we should force `populate()` regardless of status
	if(child_type_id.empty() && req_status(Req::DataNode) == ReqStatus::OK)
		return data_;
	if(const auto B = bridge()) {
		if(!data_) data_ = std::make_shared<node>();
		auto err = B->populate(data_, child_type_id);
		if(err.code == obj_fully_loaded)
			rs_reset(Req::Data, ReqReset::IfNeq, ReqStatus::OK, ReqStatus::Busy);
		return err.ok() ?
			result_or_err<sp_node>(data_) : tl::make_unexpected(std::move(err));
	}
	return tl::make_unexpected(Error::NoFusionBridge);
}

/*-----------------------------------------------------------------------------
 *  fusion_link actor
 *-----------------------------------------------------------------------------*/
// both Data & DataNode executes with `HasDataCache` flag set
auto fusion_link_actor::data_ex(obj_processor_f cb, ReqOpts opts) -> void {
	request_impl(
		*this, Req::Data, opts | ReqOpts::HasDataCache | ReqOpts::Detached,
		[Limpl = pimpl_] { return std::static_pointer_cast<fusion_link_impl>(Limpl)->data(); },
		std::move(cb)
	);
}

// `data_node` just calls `populate`
auto fusion_link_actor::data_node_ex(node_processor_f cb, ReqOpts opts) -> void {
	request_impl(
		*this, Req::DataNode, opts | ReqOpts::HasDataCache | ReqOpts::Detached,
		[Limpl = pimpl_] { return std::static_pointer_cast<fusion_link_impl>(Limpl)->populate(); },
		std::move(cb)
	);
}

auto fusion_link_actor::make_typed_behavior() -> typed_behavior {
	return first_then_second(typed_behavior_overload{
		// add handler to invoke populate with specified child type
		[=](a_flnk_populate, std::string child_type_id, bool wait_if_busy)
		-> caf::result< result_or_errbox<sp_node> > {
			auto res = make_response_promise< result_or_errbox<sp_node> >();
			request_impl(
				*this, Req::DataNode,
				ReqOpts::Detached | ReqOpts::HasDataCache |
					(wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy),
				[&I = fimpl(), cti = std::move(child_type_id)] {
					return I.populate(cti);
				},
				[=](result_or_errbox<sp_node> N) mutable { res.deliver(std::move(N)); }
			);
			return res;
		},

		// [NOTE] idea is to delegate delivery to parent actor instead of calling
		[=](a_flnk_bridge) -> caf::result<sp_fusion> {
			if(fimpl().bridge_) return fimpl().bridge_;
			// try to look up in parent link
			if(auto parent = impl.owner_.lock()) {
				if(auto phandle = fusion_link{ parent->handle() })
					return delegate(link::actor(phandle), a_flnk_bridge());
			}
			return nullptr;
		},

		[=](a_flnk_bridge, sp_fusion new_bridge) { fimpl().reset_bridge(std::move(new_bridge)); },

		// easier obj ID & object type id retrival
		[&I = fimpl()](a_lnk_otid) {
			return I.data_ ? I.data_->type_id() : nil_otid;
		},
		[&I = fimpl()](a_lnk_oid) {
			return I.data_ ? I.data_->id() : nil_oid;
		},

		// fusion link always contain a node, so directly return it's GID
		[&I = fimpl()](a_node_gid) -> result_or_errbox<std::string> {
			using R = result_or_errbox<std::string>;
			return I.data_ ? R{I.data_->gid()} : tl::make_unexpected(error::quiet(Error::EmptyData));
		}
	}, super::make_typed_behavior());
}

auto fusion_link_actor::make_behavior() -> behavior_type {
	return make_typed_behavior().unbox();
}

NAMESPACE_END(blue_sky::tree)
