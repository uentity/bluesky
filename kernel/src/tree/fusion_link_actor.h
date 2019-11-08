/// @file
/// @author uentity
/// @date 15.08.2018
/// @brief Impl part of fusion_link PIMPL
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/fusion.h>
#include <bs/tree/node.h>
#include <bs/tree/errors.h>
#include "link_actor.h"

#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

NAMESPACE_BEGIN(blue_sky::tree)
NAMESPACE_BEGIN()

// treat Error::OKOK status as object is fully loaded by fusion_iface
static const auto obj_fully_loaded = make_error_code(Error::OKOK);

// actor type for async API
using flink_actor_t = caf::typed_actor<
	caf::reacts_to<a_flnk_populate, sp_clink, link::process_data_cb, std::string>
>;

NAMESPACE_END()

// forward declare actor
struct fusion_link_actor;
using bs_detail::shared;

/*-----------------------------------------------------------------------------
 *  fusion_link_impl
 *-----------------------------------------------------------------------------*/
struct BS_HIDDEN_API fusion_link_impl : public ilink_impl {
public:
	// bridge
	sp_fusion bridge_;
	// contained object
	sp_node data_;

	using super = ilink_impl;
	using super::owner_;
	using super::lock;

	fusion_link_impl(std::string name, sp_node data, sp_fusion bridge, Flags f)
		: super(std::move(name), data, f), bridge_(std::move(bridge)), data_(std::move(data))
	{}

	fusion_link_impl()
		: super()
	{}

	// search for valid (non-null) bridge up the tree
	auto bridge() const -> sp_fusion {
		auto guard = lock(shared);

		if(bridge_) return bridge_;
		// try to look up in parent link
		if(auto parent = owner_.lock()) {
			if(auto phandle = parent->handle()) {
				if(phandle->type_id() == "fusion_link")
					return std::static_pointer_cast<fusion_link>(phandle)->bridge();
			}
		}
		return nullptr;
	}

	auto reset_bridge(sp_fusion&& new_bridge) -> void {
		auto guard = lock();
		bridge_ = std::move(new_bridge);
	}

	// implement `data`
	auto data() -> result_or_err<sp_obj> override {
		if(req_status(Req::Data) == ReqStatus::OK)
			return data_;
		if(const auto B = bridge()) {
			auto err = B->pull_data(data_);
			if(err.code == obj_fully_loaded)
				rs_reset(Req::DataNode, ReqReset::IfNeq, ReqStatus::OK, ReqStatus::Busy);
			return err.ok() ? result_or_err<sp_obj>(data_) : tl::make_unexpected(std::move(err));
		}
		return tl::make_unexpected(error{Error::NoFusionBridge});
	}

	inline auto spawn_actor(std::shared_ptr<link_impl> limpl) const -> caf::actor override;
};

/*-----------------------------------------------------------------------------
 *  fusion_link_actor
 *-----------------------------------------------------------------------------*/
struct BS_HIDDEN_API fusion_link_actor : public link_actor {
	using super = link_actor;

	using super::super;

	// implement populate with specified child type
	auto populate(const std::string& child_type_id = "", bool wait_if_busy = true)
	-> result_or_err<sp_node> {
		const auto populate_impl = [&child_type_id](fusion_link_impl* L) -> result_or_err<sp_node> {
			// assume that if `child_type_id` is nonepmty,
			// then we should force `populate()` regardless of status
			if(child_type_id.empty() && L->req_status(Req::DataNode) == ReqStatus::OK)
				return L->data_;
			if(const auto B = L->bridge()) {
				auto err = B->populate(L->data_, child_type_id);
				if(err.code == obj_fully_loaded)
					L->rs_reset(Req::Data, ReqReset::IfNeq, ReqStatus::OK, ReqStatus::Busy);
				return err.ok() ?
					result_or_err<sp_node>(L->data_) : tl::make_unexpected(std::move(err));
			}
			return tl::make_unexpected(Error::NoFusionBridge);
		};

		return detail::link_invoke(
			static_cast<fusion_link_impl*>(pimpl_.get()),
			[&populate_impl](fusion_link_impl* L) { return populate_impl(L); },
			impl.status_[1], wait_if_busy,
			// send status changed message
			function_view{ [this](ReqStatus prev_v, ReqStatus new_v) {
				if(prev_v != new_v) send(impl.self_grp, a_lnk_status(), a_ack(), new_v, prev_v);
			} }
		);
	}

	// `data_node` just calls `populate`
	auto data_node() -> result_or_err<sp_node> override {
		return populate();
	}

	auto make_behavior() -> behavior_type override {
		auto L = static_cast<fusion_link_impl*>(pimpl_.get());
		return caf::message_handler {
			// add handler to invoke populate with specified child type
			[this](a_flnk_populate, const std::string& child_type_id, bool wait_if_busy)
			-> result_or_errbox<sp_node> {
				return populate(child_type_id, wait_if_busy);
			},

			// easier obj ID & object type id retrival
			[L](a_lnk_otid) {
				return L->data_ ? L->data_->type_id() : type_descriptor::nil().name;
			},
			[L](a_lnk_oid) {
				return L->data_ ? L->data_->id() : nil_oid;
			}
		}.or_else(super::make_behavior());
	}
};

auto fusion_link_impl::spawn_actor(std::shared_ptr<link_impl> limpl) const -> caf::actor {
	return spawn_lactor<fusion_link_actor>(std::move(limpl));
}

NAMESPACE_END(blue_sky::tree)
