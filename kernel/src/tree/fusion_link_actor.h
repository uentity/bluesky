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

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <mutex>

NAMESPACE_BEGIN(blue_sky::tree)
NAMESPACE_BEGIN()

// treat Error::OKOK status as object is fully loaded by fusion_iface
static const auto obj_fully_loaded = make_error_code(Error::OKOK);

// actor type for async API
using flink_actor_t = caf::typed_actor<
	caf::reacts_to<a_flnk_populate, sp_clink, link::process_data_cb, std::string>
>;

NAMESPACE_END()

class fusion_link_actor : public ilink_actor {
public:
	// bridge
	sp_fusion bridge_;
	// contained object
	sp_node data_;
	// sync mt access
	std::mutex solo_;

	using super = ilink_actor;
	using super::owner_;
	
	fusion_link_actor(caf::actor_config& cfg, std::string name, sp_node data, sp_fusion bridge, Flags f)
		: super(cfg, std::move(name), data, f), bridge_(std::move(bridge)), data_(std::move(data))
	{}

	// search for valid (non-null) bridge up the tree
	auto bridge() const -> sp_fusion {
		if(bridge_) return bridge_;
		// try to look up in parent link
		if(auto parent = owner_.lock()) {
			if(auto phandle = parent->handle()) {
				if(phandle->type_id() == "fusion_link") {
					return std::static_pointer_cast<fusion_link>(phandle)->pimpl()->bridge();
				}
			}
		}
		return nullptr;
	}

	auto reset_bridge(sp_fusion&& new_bridge) -> void {
		std::lock_guard<std::mutex> play_solo(solo_);
		bridge_ = std::move(new_bridge);
	}

	// implement populate with specified child type
	auto populate(const std::string& child_type_id = "", bool wait_if_busy = true)
	-> result_or_err<sp_node> {
		const auto populate_impl = [this, &child_type_id]() -> result_or_err<sp_node> {
			// assume that if `child_type_id` is nonepmty,
			// then we should force `populate()` regardless of status
			if(child_type_id.empty() && req_status(Req::DataNode) == ReqStatus::OK)
				return data_;
			if(const auto B = bridge()) {
				auto err = B->populate(data_, child_type_id);
				if(err.code == obj_fully_loaded)
					rs_reset_if_neq(Req::Data, ReqStatus::Busy, ReqStatus::OK);
				return err.ok() ?
					result_or_err<sp_node>(data_) : tl::make_unexpected(std::move(err));
			}
			return tl::make_unexpected(Error::NoFusionBridge);
		};

		return detail::link_invoke(
			this,
			[&populate_impl](fusion_link_actor*) { return populate_impl(); },
			status_[1], wait_if_busy,
			// send status changed message
			function_view{ [this](ReqStatus prev_v, ReqStatus new_v) {
				if(prev_v != new_v) send(self_grp, a_lnk_status(), a_ack(), new_v, prev_v);
			} }
		);
	}

	// implement `data`
	auto data() -> result_or_err<sp_obj> override {
		if(req_status(Req::Data) == ReqStatus::OK)
			return data_;
		if(const auto B = bridge()) {
			auto err = B->pull_data(data_);
			if(err.code == obj_fully_loaded)
				rs_reset_if_neq(Req::DataNode, ReqStatus::Busy, ReqStatus::OK);
			return err.ok() ? result_or_err<sp_obj>(data_) : tl::make_unexpected(std::move(err));
		}
		return tl::make_unexpected(error{Error::NoFusionBridge});
	}

	// `data_node` just calls `populate`
	auto data_node() -> result_or_err<sp_node> override {
		return populate();
	}

	behavior_type make_behavior() override {
		return caf::message_handler {
			// add handler to invoke populate with specified child type
			[this](a_flnk_populate, const std::string& child_type_id, bool wait_if_busy)
			-> result_or_errbox<sp_node> {
				return populate(child_type_id, wait_if_busy);
			},

			// easier obj ID & object type id retrival
			[this](a_lnk_otid) {
				return data_ ? data_->type_id() : type_descriptor::nil().name;
			},
			[this](a_lnk_oid) {
				return data_ ? data_->id() : to_string(boost::uuids::nil_uuid());
			}
		}.or_else(super::make_behavior());
	}
};

NAMESPACE_END(blue_sky::tree)
