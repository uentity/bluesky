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
#include <bs/atoms.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>
#include <bs/detail/async_api_mixin.h>

#include <caf/all.hpp>

#include <mutex>

//CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::link::process_data_cb)

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

namespace {
// treat Error::OKOK status as object is fully loaded by fusion_iface
static const auto obj_fully_loaded = make_error_code(Error::OKOK);

} // hidden

struct BS_HIDDEN_API fusion_link::impl : public blue_sky::detail::async_api_mixin<fusion_link::impl> {
	// bridge
	sp_fusion bridge_;
	// contained object
	sp_node data_;
	// sync mt access
	std::mutex solo_;

	// ctor
	impl(sp_fusion&& bridge, sp_node&& data) :
		bridge_(std::move(bridge)), data_(std::move(data))
	{}

	auto reset_bridge(sp_fusion&& new_bridge) -> void {
		std::lock_guard<std::mutex> play_solo(solo_);
		bridge_ = std::move(new_bridge);
	}

	// implement populate with specified child type
	static auto populate(
		const fusion_link* lnk, const std::string& child_type_id = ""
	) -> result_or_err<sp_node> {
		// assume that if `child_type_id` is nonepmty,
		// then we should force `populate()` regardless of status
		if(child_type_id.empty() && lnk->req_status(Req::DataNode) == ReqStatus::OK) {
			return lnk->pimpl_->data_;
		}
		if(const auto B = lnk->bridge()) {
			auto err = B->populate(lnk->pimpl_->data_, child_type_id);
			if(err.code == obj_fully_loaded)
				lnk->rs_reset_if_neq(Req::Data, ReqStatus::Busy, ReqStatus::OK);
			return err.ok() ?
				result_or_err<sp_node>(lnk->pimpl_->data_) : tl::make_unexpected(std::move(err));
		}
		return tl::make_unexpected(Error::NoFusionBridge);
	}

	///////////////////////////////////////////////////////////////////////////////
	//  async API
	//
	// actor type for async API
	using actor_t = caf::typed_actor<
		caf::reacts_to<flnk_populate_atom, sp_clink, link::process_data_cb, std::string, bool>
	>;

	// async API actor handle
	actor_t actor_;
	auto actor() const -> const actor_t& { return actor_; }

	// behaviour
	static auto async_api(actor_t::pointer self) -> actor_t::behavior_type {
		using cb_arg = result_or_err<sp_clink>;

		return {
			[](flnk_populate_atom, const sp_clink& lnk, const process_data_cb& f,
				const std::string& obj_type_id, bool wait_if_busy
			) {
				f(std::static_pointer_cast<const fusion_link>(lnk)->populate(
					obj_type_id, wait_if_busy
				), lnk);
			}
		};
	}
};

NAMESPACE_END(blue_sky) NAMESPACE_END(tree)

