/// @file
/// @author uentity
/// @date 14.08.2018
/// @brief Link-related implementation details
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "link_invoke.h"
#include <bs/tree/node.h>
#include <bs/atoms.h>
#include <bs/detail/async_api_mixin.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include <caf/all.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::link::process_data_cb)

NAMESPACE_BEGIN(blue_sky::tree)

using namespace tree::detail;

using id_type = link::id_type;
using Flags = link::Flags;

/*-----------------------------------------------------------------------------
 *  hidden details
 *-----------------------------------------------------------------------------*/
namespace {

// global random UUID generator for BS links
static boost::uuids::random_generator gen;

// link's actor type for async API
using link_actor_t = caf::typed_actor<
	caf::reacts_to<lnk_data_atom, sp_clink, link::process_data_cb>,
	caf::reacts_to<lnk_dnode_atom, sp_clink, link::process_data_cb>
>;

} // eof hidden namespace

/*-----------------------------------------------------------------------------
 *  link::impl
 *-----------------------------------------------------------------------------*/
struct BS_HIDDEN_API link::impl : public blue_sky::detail::anon_async_api_mixin<link_actor_t> {
	id_type id_;
	std::string name_;
	Flags flags_;
	/// owner node
	std::weak_ptr<node> owner_;
	/// status of operations
	status_handle status_[2];
	// sync access to link's essentail data
	std::mutex solo_;

	impl(std::string&& name, Flags f)
		: anon_async_api_mixin(async_behavior), id_(gen()), name_(std::move(name)), flags_(f)
	{}

	auto rename_silent(std::string&& new_name) -> void {
		solo_.lock();
		name_ = std::move(new_name);
		solo_.unlock();
	}

	auto rename(std::string&& new_name) -> void {
		rename_silent(std::move(new_name));
		// [TODO] send message instead
		if(auto O = owner_.lock()) {
			O->on_rename(id_);
		}
	}

	auto req_status(Req request) const -> ReqStatus {
		const auto i = (unsigned)request;
		if(i < 2){
			return status_[i].value;
		}
		return ReqStatus::Void;
	}

	auto rs_reset(Req request, ReqStatus new_rs) {
		const auto i = (unsigned)request;
		if(i >= 2) return ReqStatus::Error;

		// atomic set value
		auto S = scope_atomic_flag(status_[i].flag);
		const auto self = status_[i].value;
		status_[i].value = new_rs;
		return self;
	}

	auto rs_reset_if_eq(Req request, ReqStatus self_rs, ReqStatus new_rs) {
		const auto i = (unsigned)request;
		if(i >= 2) return ReqStatus::Error;

		// atomic set value
		auto S = scope_atomic_flag(status_[i].flag);
		const auto self = status_[i].value;
		if(status_[i].value == self_rs) status_[i].value = new_rs;
		return self;
	}

	auto rs_reset_if_neq(Req request, ReqStatus self_rs, ReqStatus new_rs) {
		const auto i = (unsigned)request;
		if(i >= 2) return ReqStatus::Error;

		// atomic set value
		auto S = scope_atomic_flag(status_[i].flag);
		const auto self = status_[i].value;
		if(status_[i].value != self_rs) status_[i].value = new_rs;
		return self;
	}

	///////////////////////////////////////////////////////////////////////////////
	//  async API behavior
	//
	static auto async_behavior(link_actor_t::pointer self) -> link_actor_t::behavior_type {
		return {
			[](lnk_data_atom, const sp_clink& lnk, const process_data_cb& f) {
				f(lnk->data_ex(true), lnk);
			},
			[](lnk_dnode_atom, const sp_clink& lnk, const process_data_cb& f) {
				f(lnk->data_node_ex(true), lnk);
			}
		};
	}
};

NAMESPACE_END(blue_sky::tree)
