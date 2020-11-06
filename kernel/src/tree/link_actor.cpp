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
#include <bs/serialize/propdict.h>

#include "link_actor.h"
#include "request_impl.h"
#include "../objbase_actor.h"
#include "../serialize/tree_impl.h"

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)
using namespace allow_enumops;
using namespace kernel::radio;
using namespace std::chrono_literals;

[[maybe_unused]] auto adbg_impl(link_actor* A) -> caf::actor_ostream {
	return caf::aout(A) << "[L:" << A->impl.type_id() << "] [" << to_string(A->impl.id_) <<
		"] [" << A->impl.name_ << "]: ";
}

// helper to apply data transactions
template<typename... Ts>
static auto do_data_apply(link_actor* LA, transaction_t<tr_result, Ts...> tr) {
	auto res = LA->make_response_promise<tr_result::box>();
	LA->request(LA->actor(), caf::infinite, a_data(), true)
	.then([=, tr = std::move(tr)](obj_or_errbox maybe_obj) mutable {
		if(maybe_obj) {
			// always send closed transaction
			transaction tr_ = [&] {
				if constexpr(sizeof...(Ts) > 0)
					return (*maybe_obj)->make_transaction(std::move(tr));
				else
					return std::move(tr);
			}();
			LA->request(maybe_obj.value()->actor(), caf::infinite, a_apply(), std::move(tr_))
			.then([=](tr_result::box tres) mutable {
				res.deliver(std::move(tres));
			});
		}
		else
			res.deliver(pack(tr_result{std::move(maybe_obj).error()}));
	});
	return res;
}

/*-----------------------------------------------------------------------------
 *  link
 *-----------------------------------------------------------------------------*/
link_actor::link_actor(caf::actor_config& cfg, caf::group lgrp, sp_limpl Limpl) :
	super(cfg), pimpl_(std::move(Limpl)), impl([this]() -> link_impl& {
		if(!pimpl_) throw error{"link actor: bad (null) link impl passed"};
		return *pimpl_;
	}()), ropts_{ReqOpts::WaitIfBusy, ReqOpts::WaitIfBusy}
{
	// remember link's local group
	impl.home = std::move(lgrp);
	if(impl.home)
		adbg(this) << "joined self group " << std::endl;

	// exit after kernel
	KRADIO.register_citizen(this);

	// prevent termination in case some errors happens in group members
	// for ex. if they receive unexpected messages (translators normally do)
	set_error_handler([this](caf::error er) {
		switch(static_cast<caf::sec>(er.code())) {
		case caf::sec::unexpected_message :
		case caf::sec::request_timeout :
		case caf::sec::request_receiver_down :
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

	[=](a_home_id) { return std::string(impl.home_id()); },

	[=](a_impl) -> sp_limpl {
		return pimpl_;
	},

	// subscribe events listener
	[=](a_subscribe, const caf::actor& baby) {
		// remember baby & ensure it's alive
		return delegate(caf::actor_cast<ev_listener_actor_type>(baby), a_hi(), impl.home);
	},

	// apply link transaction
	[=](a_apply, simple_transaction tr) -> error::box {
		return tr_eval(std::move(tr));
	},

	[=](a_apply, link_transaction tr) -> error::box {
		return tr_eval(std::move(tr), bare_link(pimpl_));
	},

	// apply data transactions
	[=](a_apply, a_data, transaction tr) -> caf::result<tr_result::box> {
		return do_data_apply(this, std::move(tr));
	},
	[=](a_apply, a_data, obj_transaction tr) -> caf::result<tr_result::box> {
		return do_data_apply(this, std::move(tr));
	},

	// get id
	[=](a_lnk_id) -> lid_type {
		adbg(this) << "<- a_lnk_id: " << to_string(impl.id_) << std::endl;
		return impl.id_;
	},

	// get oid
	[=](a_lnk_oid) -> std::string {
		// [NOTE] assume that getting data if status == OK is fast
		auto res = std::string{};
		request_data(
			*this, ReqOpts::ErrorIfNOK | ReqOpts::DirectInvoke,
			[&](obj_or_errbox obj) mutable {
				res = obj ? obj.value()->id() : nil_oid;
				adbg(this) << "<- a_lnk_oid: " << res << std::endl;
			}
		);
		return res;
	},

	// get object type_id
	[=](a_lnk_otid) -> std::string {
		// [NOTE] assume that getting data if status == OK is fast
		auto res = std::string{};
		request_data(
			*this, ReqOpts::ErrorIfNOK | ReqOpts::DirectInvoke,
			[&](obj_or_errbox obj) mutable {
				res = obj ? obj.value()->type_id() : nil_otid;
				adbg(this) << "<- a_lnk_otid: " << res << std::endl;
			}
		);
		return res;
	},

	// get name
	[=](a_lnk_name) -> std::string {
		adbg(this) << "<- a_lnk_name: " << impl.name_ << std::endl;
		return impl.name_;
	},

	// rename
	[=](a_lnk_rename, std::string new_name) -> caf::result<std::size_t> {
		adbg(this) << "<- a_lnk_rename " << impl.name_ << " -> " << new_name << std::endl;

		// if link belongs to some node, use node::rename (to update index)
		if(auto master = impl.owner()) {
			adbg(this) << "-> delegated to master" << std::endl;
			return delegate(master.actor(), a_lnk_rename(), impl.id_, std::move(new_name));
		}
		// otherwise rename directly
		else {
			impl.rename(std::move(new_name));
			return std::size_t{1};
		}
	},

	// get status
	[=](a_lnk_status, Req req) -> ReqStatus { return pimpl_->req_status(req); },

	// change status
	[=](a_lnk_status, Req req, ReqReset cond, ReqStatus new_rs, ReqStatus prev_rs) -> ReqStatus {
		adbg(this) << "<- a_lnk_status: " << to_string(req) << " " <<
			to_string(prev_rs) << "->" << to_string(new_rs) << std::endl;
		return rs_reset(req, cond, new_rs, prev_rs);
	},

	// just send notification about just changed status
	[=](a_lnk_status, Req req, ReqStatus new_rs, ReqStatus prev_rs) {
		impl.send_home<high_prio>(this, a_ack(), a_lnk_status(), req, new_rs, prev_rs);
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
			to_string(impl.req_status(Req::Data)) << "," << to_string(impl.req_status(Req::DataNode)) << std::endl;

		auto res = make_response_promise< obj_or_errbox >();
		request_data(
			*this, ropts_.data | (wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy),
			[=](obj_or_errbox obj) mutable { res.deliver(std::move(obj)); }
		);
		return res;
	},

	// get data node
	[=](a_data_node, bool wait_if_busy) -> caf::result< node_or_errbox > {
		adbg(this) << "<- a_data_node, status = " <<
			to_string(impl.req_status(Req::Data)) << "," << to_string(impl.req_status(Req::DataNode)) << std::endl;

		auto res = make_response_promise< node_or_errbox >();
		request_data_node(
			*this, ropts_.data_node | (wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy),
			[=](node_or_errbox N) mutable { res.deliver(std::move(N)); }
		);
		return res;
	},

	// noop - implement in derived links
	[=](a_lazy, a_load) { return false; }
}; }

auto link_actor::make_typed_behavior() -> typed_behavior {
	return first_then_second(make_ack_behavior(), make_primary_behavior());
}

auto link_actor::make_behavior() -> behavior_type {
	return make_typed_behavior().unbox();
}

/*-----------------------------------------------------------------------------
 *  cached_link_actor
 *-----------------------------------------------------------------------------*/
cached_link_actor::cached_link_actor(caf::actor_config& cfg, caf::group self_grp, sp_limpl Limpl) :
	link_actor(cfg, std::move(self_grp), std::move(Limpl))
{
	ropts_ = {ReqOpts::HasDataCache, ReqOpts::HasDataCache};
}

auto cached_link_actor::make_typed_behavior() -> typed_behavior {
	return first_then_second( typed_behavior_overload{
		// OID & object type id are retrieved from cached value
		[=](a_lnk_otid) -> std::string {
			if(auto obj = impl.data(unsafe))
				return obj->type_id();
			return nil_otid;
		},

		[=](a_lnk_oid) -> std::string {
			if(auto obj = impl.data(unsafe))
				return obj->id();
			return nil_otid;
		},

		// modifies beahvior s.t. next Data request will send `a_delay_load` to stored object first
		[=](a_lazy, a_load) {
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
					request(objbase_actor::actor(*obj), caf::infinite, a_load(), std::move(obj))
					.then([=, orig_me = std::move(orig_me)](error::box er) mutable {
						// if error happened - deliver it
						if(er.ec) res.deliver(R{ tl::unexpect, std::move(er) });
						// otherwise, make data request to self (orig_me)
						request(caf::actor_cast<actor_type>(this), caf::infinite, req_t(), true)
						.then(
							[=](R req_res) mutable { res.deliver(std::move(req_res)); },
							[=](const caf::error& er) mutable {
								res.deliver(R{ tl::unexpect, forward_caf_error(er) });
							}
						);
					});
					return res;
				};
			};

			// setup new behavior for Data & DataNode requests
			become(caf::message_handler{
				load_then_answer(a_data()), load_then_answer(a_data_node()),

				// OID & OTID will trigger load
				[=](a_lnk_otid) -> caf::result<std::string> {
					auto res = make_response_promise<std::string>();
					request(caf::actor_cast<caf::actor>(this), caf::infinite, a_data(), true)
					.then([=](const obj_or_errbox& maybe_obj) mutable {
						if(maybe_obj) res.deliver((*maybe_obj)->type_id());
						res.deliver(std::string{nil_otid});
					});
					return res;
				},

				[=](a_lnk_oid) -> caf::result<std::string> {
					auto res = make_response_promise<std::string>();
					request(caf::actor_cast<caf::actor>(this), caf::infinite, a_data(), true)
					.then([=](const obj_or_errbox& maybe_obj) mutable {
						if(maybe_obj) res.deliver((*maybe_obj)->id());
						res.deliver(std::string{nil_oid});
					});
					return res;
				},
			}.or_else(orig_me));
			return true;
		}
	}, super::make_typed_behavior() );
}

auto cached_link_actor::make_behavior() -> behavior_type {
	return make_typed_behavior().unbox();
}

NAMESPACE_END(blue_sky::tree)
