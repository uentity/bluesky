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
#include "request_impl.h"

OMIT_OBJ_SERIALIZATION
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::sp_fusion)

NAMESPACE_BEGIN(blue_sky::tree)
using namespace allow_enumops;
using bs_detail::shared;

// forward declare actor
struct fusion_link_actor;

/*-----------------------------------------------------------------------------
 *  fusion_link_impl
 *-----------------------------------------------------------------------------*/
struct BS_HIDDEN_API fusion_link_impl : public ilink_impl {
	// treat Error::OKOK status as object is fully loaded by fusion_iface
	inline static const auto obj_fully_loaded = make_error_code(Error::OKOK);
	// bridge
	sp_fusion bridge_;
	// contained object
	sp_node data_;

	using actor_type = fusion_link::actor_type;

	using super = ilink_impl;
	using super::owner_;

	fusion_link_impl(std::string name, sp_node data, sp_fusion bridge, Flags f) :
		super(std::move(name), data, f), bridge_(std::move(bridge)), data_(std::move(data))
	{}

	fusion_link_impl() :
		super()
	{}

	// search for valid (non-null) bridge up the tree
	auto bridge() const -> sp_fusion {
		if(bridge_) return bridge_;
		// try to look up in parent link
		if(auto parent = owner_.lock()) {
			if(auto phandle = fusion_link{ parent->handle() })
				return phandle.bridge();
		}
		return nullptr;
	}

	auto reset_bridge(sp_fusion&& new_bridge) -> void {
		bridge_ = std::move(new_bridge);
	}

	// implement `data`
	auto data() -> result_or_err<sp_obj> override {
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
	auto data(unsafe_t) -> sp_obj override {
		return data_;
	}

	// populate with specified child type
	auto populate(const std::string& child_type_id = "", bool wait_if_busy = true)
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

	auto spawn_actor(std::shared_ptr<link_impl> limpl) const -> caf::actor override;

	auto clone(bool deep = false) const -> sp_limpl override;

	auto propagate_handle(const link&) -> result_or_err<sp_node> override;

	LIMPL_TYPE_DECL
};

/*-----------------------------------------------------------------------------
 *  fusion_link_actor
 *-----------------------------------------------------------------------------*/
struct BS_HIDDEN_API fusion_link_actor : public cached_link_actor {
	using super = cached_link_actor;
	using super::super;

	using actor_type = fusion_link_impl::actor_type;
	using typed_behavior = actor_type::behavior_type;
	// part of behavior overloaded/added from super actor type
	using typed_behavior_overload = caf::typed_behavior<
		caf::replies_to<a_flnk_populate, std::string, bool>::with<result_or_errbox<sp_node>>,
		caf::replies_to<a_flnk_bridge>::with<sp_fusion>,
		caf::reacts_to<a_flnk_bridge, sp_fusion>,

		// get pointee OID
		caf::replies_to<a_lnk_oid>::with<std::string>,
		// get pointee type ID
		caf::replies_to<a_lnk_otid>::with<std::string>,
		// get pointee node group ID
		caf::replies_to<a_node_gid>::with<result_or_errbox<std::string>>,
		// get data cache
		caf::replies_to<a_lnk_dcache>::with<sp_obj>
	>;

	auto fimpl() -> fusion_link_impl& { return static_cast<fusion_link_impl&>(impl); }

	// both Data & DataNode executes with `HasDataCache` flag set
	auto data_ex(obj_processor_f cb, ReqOpts opts) -> void override {
		request_impl(
			*this, Req::Data, opts | ReqOpts::HasDataCache | ReqOpts::Detached,
			[Limpl = pimpl_] { return std::static_pointer_cast<fusion_link_impl>(Limpl)->data(); },
			std::move(cb)
		);
	}

	// `data_node` just calls `populate`
	auto data_node_ex(node_processor_f cb, ReqOpts opts) -> void override {
		request_impl(
			*this, Req::DataNode, opts | ReqOpts::HasDataCache | ReqOpts::Detached,
			[Limpl = pimpl_] { return std::static_pointer_cast<fusion_link_impl>(Limpl)->populate(); },
			std::move(cb)
		);
	}

	auto make_typed_behavior() -> typed_behavior {
		return first_then_second( typed_behavior_overload{
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

			// direct return cached data
			[=](a_lnk_dcache) -> sp_obj { return fimpl().data_; },

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

	auto make_behavior() -> behavior_type override {
		return make_typed_behavior().unbox();
	}
};

NAMESPACE_END(blue_sky::tree)
