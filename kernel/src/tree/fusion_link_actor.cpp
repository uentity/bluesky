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

#include <bs/kernel/types_factory.h>

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  fusion_link impl
 *-----------------------------------------------------------------------------*/
fusion_link_impl::fusion_link_impl(std::string name, sp_obj data, sp_fusion bridge, Flags f) :
	super(std::move(name), data, f), data_(data ? std::move(data) : std::make_shared<objnode>()),
	bridge_(std::move(bridge))
{}

fusion_link_impl::fusion_link_impl() :
	super(), data_(std::make_shared<objnode>())
{}

auto fusion_link_impl::spawn_actor(std::shared_ptr<link_impl> limpl) const -> caf::actor {
	return spawn_lactor<fusion_link_actor>(std::move(limpl));
}

auto fusion_link_impl::clone(bool deep) const -> sp_limpl {
	return std::make_shared<fusion_link_impl>(
		name_,
		deep ? kernel::tfactory::clone_object(data_) : data_,
		bridge_, flags_
	);
}

// search for valid (non-null) bridge up the tree
auto fusion_link_impl::bridge() const -> sp_fusion {
	{
		auto solo = std::shared_lock{ bridge_guard_ };
		if(bridge_) return bridge_;
	}
	// try to look up in parent link
	if(auto parent = owner()) {
		if(auto phandle = fusion_link{ parent.handle() })
			return phandle.bridge();
	}
	return nullptr;
}

auto fusion_link_impl::reset_bridge(sp_fusion&& new_bridge) -> void {
	auto solo = std::lock_guard{ bridge_guard_ };
	bridge_ = std::move(new_bridge);
}

// request data via bridge
auto fusion_link_impl::data() -> obj_or_err {
	if(req_status(Req::Data) != ReqStatus::OK) {
		const auto B = bridge();
		if(!B) return unexpected_err(Error::NoFusionBridge);

		if(auto err = B->pull_data(data_, super_engine()))
			return tl::make_unexpected(std::move(err));
	}
	return data_;
}

// unsafe version returns cached value
auto fusion_link_impl::data(unsafe_t) const -> sp_obj {
	return data_;
}

// populate with specified child type
auto fusion_link_impl::populate(const std::string& child_type_id) -> node_or_err {
	// assume that if `child_type_id` is nonepmty,
	// then we should force `populate()` regardless of status
	if(req_status(Req::DataNode) != ReqStatus::OK || !child_type_id.empty()) {
		const auto B = bridge();
		if(!B) return unexpected_err(Error::NoFusionBridge);

		if(auto err = B->populate(data_, super_engine(), child_type_id))
			return tl::make_unexpected(std::move(err));
	}
	return data_->data_node();
}

/*-----------------------------------------------------------------------------
 *  fusion_link actor
 *-----------------------------------------------------------------------------*/
auto fusion_link_actor::make_ropts(Req r) -> ReqOpts {
	// both Data & DataNode execute with `HasDataCache` flag set
	auto opts = (r == Req::Data ? ropts_.data : ropts_.data_node) |
		ReqOpts::HasDataCache | ReqOpts::Detached;
	if(auto B = fimpl().bridge())
		error::eval_safe([&] {
			if(B->is_uniform(fimpl().data_)) opts |= ReqOpts::Uniform;
		});
	return opts;
}

auto fusion_link_actor::make_typed_behavior() -> typed_behavior {
	return first_then_second(typed_behavior_overload{
		// invoke populate with specified child type
		[=](a_flnk_populate, std::string child_type_id, bool wait_if_busy) -> caf::result<node_or_errbox> {
			return request_data_impl(
				*this, Req::DataNode,
				make_ropts(Req::DataNode) | (wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy),
				[Limpl = pimpl_, ctid = std::move(child_type_id)] {
					return static_cast<fusion_link_impl&>(*Limpl).populate(ctid);
				}
			);
		},

		// DataNode request = populate with empty child_type_id
		[=](a_data_node, bool wait_if_busy) -> caf::result<node_or_errbox> {
			return delegate(
				caf::actor_cast<fusion_link::actor_type>(this),
				a_flnk_populate(), std::string{}, wait_if_busy
			);
		},

		// Data
		[=](a_data, bool wait_if_busy) -> caf::result<obj_or_errbox> {
			return request_data(
				*this, make_ropts(Req::Data) | (wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy)
			);
		},

		// bridge manip
		[=](a_flnk_bridge) -> sp_fusion { return fimpl().bridge(); },
		[=](a_flnk_bridge, sp_fusion new_bridge) { fimpl().reset_bridge(std::move(new_bridge)); }

	}, super::make_typed_behavior());
}

auto fusion_link_actor::make_behavior() -> behavior_type {
	return make_typed_behavior().unbox();
}

NAMESPACE_END(blue_sky::tree)
