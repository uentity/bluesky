/// @file
/// @author uentity
/// @date 09.07.2019
/// @brief Base link actor implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include <bs/kernel/tools.h>
#include <bs/kernel/config.h>
#include <bs/kernel/radio.h>

#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include "link_actor.h"
#include "../objbase_actor.h"

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)
using namespace allow_enumops;
using namespace kernel::radio;
using namespace std::chrono_literals;

[[maybe_unused]] auto adbg_impl(link_actor* A) -> caf::actor_ostream {
	return caf::aout(A) << "[L] [" << to_string(A->impl.id_) <<
		"] [" << A->impl.name_ << "]: ";
}

/*-----------------------------------------------------------------------------
 *  link
 *-----------------------------------------------------------------------------*/
link_actor::link_actor(caf::actor_config& cfg, caf::group lgrp, sp_limpl Limpl)
	: super(cfg), pimpl_(std::move(Limpl)), impl([this]() -> link_impl& {
		if(!pimpl_) throw error{"link actor: bad (null) link impl passed"};
		return *pimpl_;
	}())
{
	// remember link's local group
	impl.home = std::move(lgrp);
	if(impl.home)
		adbg(this) << "joined self group " << impl.home.get()->identifier() << std::endl;

	// exit after kernel
	KRADIO.register_citizen(this);

	// prevent termination in case some errors happens in group members
	// for ex. if they receive unexpected messages (translators normally do)
	set_error_handler([this](caf::error er) {
		switch(static_cast<caf::sec>(er.code())) {
		case caf::sec::unexpected_message :
			break;
		default:
			default_error_handler(this, er);
		}
	});

	// completely ignore unexpected messages without error backpropagation
	set_default_handler([](auto*, auto&) -> caf::result<caf::message> {
		return caf::none;
	});
}

link_actor::~link_actor() = default;

auto link_actor::on_exit() -> void {
	adbg(this) << "dies" << std::endl;

	// be polite with everyone
	goodbye();
	// force release strong ref to link's impl
	pimpl_->release_factors();
	pimpl_.reset();

	KRADIO.release_citizen(this);
}

auto link_actor::goodbye() -> void {
	adbg(this) << "goodbye" << std::endl;
	if(impl.home) {
		// say goodbye to self group
		send(impl.home, a_bye());
		leave(impl.home);
		adbg(this) << "left self group " << impl.home.get()->identifier() << std::endl;
		//	<< "\n" << kernel::tools::get_backtrace(30, 4) << std::endl;
	}
}

auto link_actor::name() const -> const char* {
	return "link_actor";
}

auto link_actor::rename(std::string new_name, bool silent) -> void {
	adbg(this) << "<- a_lnk_rename " << (silent ? "[silent]: " : ": ") << impl.name_ <<
		" -> " << new_name << std::endl;

	auto old_name = impl.name_;
	impl.name_ = std::move(new_name);
	// send rename ack message
	if(!silent)
		impl.send_home<high_prio>(this, a_ack(), a_lnk_rename(), impl.name_, std::move(old_name));
}

auto link_actor::rs_reset(Req req, ReqReset cond, ReqStatus new_rs, ReqStatus prev_rs, bool silent)
-> ReqStatus {
	return impl.rs_reset(
		req, cond, new_rs, prev_rs,
		silent ? noop :
			function_view{[=](Req req, ReqStatus new_s, ReqStatus old_s) {
				impl.send_home<high_prio>(this, a_ack(), a_lnk_status(), req, new_s, old_s);
			}}
	);
}

///////////////////////////////////////////////////////////////////////////////
//  behavior
//
auto link_actor::make_primary_behavior() -> primary_actor_type::behavior_type {
return {
	// skip `bye` message (should always come from myself)
	[=](a_bye) {
		adbg(this) << "<- a_lnk_bye " << std::endl;
		if(current_sender() != this) quit();
	},

	[=](a_home) { return impl.home; },

	[=](a_home_id) { return impl.home_id(); },

	[=](a_impl) -> sp_limpl {
		return pimpl_;
	},

	// get id
	[=](a_lnk_id) -> lid_type {
		adbg(this) << "<- a_lnk_id: " << to_string(impl.id_) << std::endl;
		return impl.id_;
	},

	// get oid
	[=](a_lnk_oid) -> std::string {
		// [NOTE] assume that if status is OK then getting data is fast (data is cached)
		auto res = std::string{};
		data_ex(
			[&](obj_or_errbox obj) mutable {
				res = obj ? obj.value()->id() : nil_oid;
				adbg(this) << "<- a_lnk_oid: " << res << std::endl;
			},
			ReqOpts::ErrorIfNOK | ReqOpts::DirectInvoke
		);
		return res;
	},

	// get object type_id
	[=](a_lnk_otid) -> std::string {
		// [NOTE] assume that if status is OK then getting data is fast (data is cached)
		auto res = std::string{};
		//auto tstart = make_timestamp();
		data_ex(
			[&](obj_or_errbox obj) mutable {
				res = obj ? obj.value()->type_id() : nil_otid;
				adbg(this) << "<- a_lnk_otid: " << res << std::endl;
			},
			ReqOpts::ErrorIfNOK | ReqOpts::DirectInvoke
		);
		return res;
	},

	// get node's group ID
	//[=](a_node_gid) -> result_or_errbox<std::string> {
	//	adbg(this) << "<- a_node_gid" << std::endl;
	//	auto res = result_or_err<std::string>{};
	//	data_node_ex(
	//		[&](node_or_errbox N) {
	//			N.map([&](const node& N) {
	//				res = N->gid();
	//			});
	//		},
	//		ReqOpts::ErrorIfNOK | ReqOpts::DirectInvoke
	//	);
	//	return res;
	//},

	// get name
	[=](a_lnk_name) -> std::string {
		adbg(this) << "<- a_lnk_name: " << impl.name_ << std::endl;
		return impl.name_;
	},

	// rename
	[=](a_lnk_rename, std::string new_name, bool silent) -> void {
		rename(std::move(new_name), silent);
	},

	// get status
	[=](a_lnk_status, Req req) -> ReqStatus { return pimpl_->req_status(req); },

	// change status
	[=](a_lnk_status, Req req, ReqReset cond, ReqStatus new_rs, ReqStatus prev_rs) -> ReqStatus {
		adbg(this) << "<- a_lnk_status: " << to_string(req) << " " <<
			to_string(prev_rs) << "->" << to_string(new_rs) << std::endl;
		return rs_reset(req, cond, new_rs, prev_rs);
	},

	// get/set flags
	[=](a_lnk_flags) { return pimpl_->flags_; },
	[=](a_lnk_flags, Flags f) { pimpl_->flags_ = f; },

	// obtain inode
	// [NOTE] assume it's a fast call, override behaviour where needed (sym_link for ex)
	[=](a_lnk_inode) -> result_or_errbox<inodeptr> {
		adbg(this) << "<- a_lnk_inode" << std::endl;
		return impl.get_inode();
	},

	// get data
	[=](a_data, bool wait_if_busy) -> caf::result< obj_or_errbox > {
		adbg(this) << "<- a_data, status = " <<
			to_string(impl.status_[0].value) << "," << to_string(impl.status_[1].value) << std::endl;

		auto res = make_response_promise< obj_or_errbox >();
		data_ex(
			[=](obj_or_errbox obj) mutable { res.deliver(std::move(obj)); },
			(wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy) //| ReqOpts::Detached
		);
		return res;
	},

	// get data node
	[=](a_data_node, bool wait_if_busy) -> caf::result< node_or_errbox > {
		adbg(this) << "<- a_data_node, status = " <<
			to_string(impl.status_[0].value) << "," << to_string(impl.status_[1].value) << std::endl;

		auto res = make_response_promise< node_or_errbox >();
		data_node_ex(
			[=](node_or_errbox N) mutable { res.deliver(std::move(N)); },
			(wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy) //| ReqOpts::Detached
		);
		return res;
	},

	// noop - implement in derived links
	[=](a_delay_load) { return false; }
}; }

auto link_actor::make_typed_behavior() -> typed_behavior {
	return first_then_second(make_ack_behavior(), make_primary_behavior());
}

auto link_actor::make_behavior() -> behavior_type {
	return make_typed_behavior().unbox();
}

/*-----------------------------------------------------------------------------
 *  fast_link_actor
 *-----------------------------------------------------------------------------*/
// for fast link we can assume that requests are invoked directly
// => override slow API to exclude extra delivery messages
auto fast_link_actor::make_typed_behavior() -> typed_behavior {
	return first_then_second( typed_behavior_overload{
		// obtain inode
		[=](a_lnk_inode) -> result_or_errbox<inodeptr> {
			adbg(this) << "<- a_lnk_inode" << std::endl;

			return impl.get_inode();
		},

		[=](a_data, bool) -> obj_or_errbox {
			adbg(this) << "<- a_data fast, status = " <<
				to_string(impl.status_[0].value) << "," << to_string(impl.status_[1].value) << std::endl;

			return pimpl_->data().and_then([](auto&& obj) {
				return obj ?
					obj_or_errbox(std::move(obj)) :
					unexpected_err_quiet(Error::EmptyData);
			});

			//auto res = obj_or_errbox{};
			//data_ex(
			//	[&](obj_or_errbox obj) { res = std::move(obj); },
			//	ReqOpts::WaitIfBusy
			//);
			//return res;
		},

		[=](a_data_node, bool) -> node_or_errbox {
			adbg(this) << "<- a_data_node fast, status = " <<
				to_string(impl.status_[0].value) << "," << to_string(impl.status_[1].value) << std::endl;

			return pimpl_->data().and_then([](const auto& obj) -> node_or_errbox {
				if(obj) {
					if(auto n = obj->data_node())
						return n;
					return unexpected_err_quiet(Error::NotANode);
				}
				return unexpected_err_quiet(Error::EmptyData);
			});

			//auto res = node_or_errbox();
			//data_node_ex(
			//	[&](node_or_errbox N) { res = std::move(N); },
			//	ReqOpts::WaitIfBusy
			//);
			//return res;
		},

		[=](a_delay_load) {
			auto orig_me = current_behavior();

			// setup request impl that invokes `a_delay_load` on object once
			const auto load_then_answer = [&](auto req) {
				using req_t = decltype(req);
				using R = std::conditional_t<std::is_same_v<req_t, a_data>, obj_or_errbox, node_or_errbox>;

				return [=](req_t, bool) mutable -> caf::result<R> {
					// this handler triggered only once
					become(orig_me);
					// get cached object
					auto obj = impl.data(unsafe);
					if(!obj) return unexpected_err_quiet(Error::EmptyData);

					auto res = make_response_promise<R>();
					auto objA = objbase_actor::actor(*obj);
					request(std::move(objA), caf::infinite, a_delay_load(), std::move(obj))
					.then([=, orig_me = std::move(orig_me)](error::box er) mutable {
						// invoke original behavior inplace
						auto m = caf::make_message(req_t(), true);
						orig_me(m)->extract(
							[&](R r) { res.deliver(std::move(r)); }
						);
					});
					return res;
				};
			};

			// setup new behavior for Data & DataNode requests 
			become(caf::message_handler{
				load_then_answer(a_data()), load_then_answer(a_data_node())
			}.or_else(orig_me));
			return true;
		}
	}, super::make_typed_behavior() );
}

auto fast_link_actor::make_behavior() -> behavior_type {
	return make_typed_behavior().unbox();
}

NAMESPACE_END(blue_sky::tree)
