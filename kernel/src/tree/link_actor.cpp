/// @file
/// @author uentity
/// @date 09.07.2019
/// @brief Base link actor implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "link_actor.h"
#include <bs/log.h>
#include <bs/kernel/tools.h>
#include <bs/kernel/config.h>
#include <bs/kernel/radio.h>

#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include <boost/uuid/uuid_generators.hpp>

#define DEBUG_ACTOR 0

#if DEBUG_ACTOR == 1
#include <caf/actor_ostream.hpp>
#endif

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;
using namespace std::chrono_literals;
using bs_detail::shared;

NAMESPACE_BEGIN()
#if DEBUG_ACTOR == 1

auto adbg(link_actor* A) -> caf::actor_ostream {
	return caf::aout(A) << "[L] [" << to_string(A->impl.id_) <<
		"] [" << A->impl.name_ << "]: ";
}

#else

constexpr auto adbg(link_actor*) { return log::D(); }

#endif
NAMESPACE_END()

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
	impl.self_grp = std::move(lgrp);
	adbg(this) << "joined self group " << impl.self_grp.get()->identifier() << std::endl;

	// on exit say goodbye to self group
	set_exit_handler([this](caf::exit_msg& er) {
		goodbye();
		default_exit_handler(this, er);
	});

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

	set_default_handler(caf::drop);
}

link_actor::~link_actor() = default;

auto link_actor::name() const -> const char* {
	return "link_actor";
}

auto link_actor::on_exit() -> void {
	adbg(this) << "dies" << std::endl;
}

auto link_actor::goodbye() -> void {
	adbg(this) << "goodbye" << std::endl;
	if(impl.self_grp) {
		// say goodbye to self group
		send(impl.self_grp, a_bye());
		leave(impl.self_grp);
		adbg(this) << "left self group " << impl.self_grp.get()->identifier() << std::endl;
		//	<< "\n" << kernel::tools::get_backtrace(30, 4) << std::endl;
	}
}

auto link_actor::rename(std::string new_name, bool silent) -> void {
	adbg(this) << "<- a_lnk_rename " << (silent ? "silent: " : "loud: ") << impl.name_ <<
		" -> " << new_name << std::endl;

	auto old_name = impl.name_;
	impl.name_ = std::move(new_name);
	// send rename ack message
	if(!silent)
		send<high_prio>(impl.self_grp, a_ack(), a_lnk_rename(), impl.name_, std::move(old_name));
}


///////////////////////////////////////////////////////////////////////////////
//  behavior
//
auto link_actor::make_behavior() -> behavior_type { return {
	/// skip `bye` message (should always come from myself)
	[this](a_bye) {
		adbg(this) << "<- a_lnk_bye" << std::endl;
	},

	/// get id
	[this](a_lnk_id) {
		adbg(this) << "<- a_lnk_id: " << to_string(impl.id_) << std::endl;
		return impl.id_;
	},

	/// get name
	[this](a_lnk_name) {
		adbg(this) << "<- a_lnk_name: " << impl.name_ << std::endl;
		return impl.name_;
	},

	/// rename
	[this](a_lnk_rename, std::string new_name, bool silent) {
		rename(std::move(new_name), silent);
	},
	// rename ack
	[this](a_ack, a_lnk_rename, std::string new_name, const std::string& old_name) {
		adbg(this) << "<- a_lnk_rename ack: " << old_name << "->" << new_name << std::endl;
		if(current_sender() != this)
			rename(std::move(new_name), true);
	},

	// get status
	[this](a_lnk_status, Req req) { return pimpl_->req_status(req); },

	// change status
	[this](a_lnk_status, Req req, ReqReset cond, ReqStatus new_rs, ReqStatus prev_rs) {
		adbg(this) << "<- a_lnk_status: " << to_string(req) << " " <<
			to_string(prev_rs) << "->" << to_string(new_rs) << std::endl;
		return impl.rs_reset(
			req, cond, new_rs, prev_rs,
			[this](Req req, ReqStatus new_s, ReqStatus old_s) {
				send<high_prio>(impl.self_grp, a_ack(), a_lnk_status(), req, new_s, old_s);
			}
		);
	},

	[this](a_ack, a_lnk_status, Req req, ReqStatus new_s, ReqStatus prev_s) {
		adbg(this) << "<- a_lnk_status ack: " << to_string(req) << " " <<
			to_string(prev_s) << "->" << to_string(new_s) << std::endl;
	},

	// get/set flags
	[this](a_lnk_flags) { return pimpl_->flags_; },
	[this](a_lnk_flags, Flags f) { pimpl_->flags_ = f; },

	// get oid
	[this](a_lnk_oid) -> std::string {
		// [NOTE] assume that if status is OK then getting data is fast (data is cached)
		auto res = std::string{};
		data_ex(
			[&](result_or_errbox<sp_obj> obj) mutable {
				res = obj ? obj.value()->id() : nil_oid;
				adbg(this) << "<- a_lnk_oid: " << res << std::endl;
			},
			ReqOpts::ErrorIfNOK | ReqOpts::DirectInvoke
		);
		return res;
	},

	// get object type_id
	[this](a_lnk_otid) -> std::string {
		// [NOTE] assume that if status is OK then getting data is fast (data is cached)
		auto res = std::string{};
		//auto tstart = make_timestamp();
		data_ex(
			[&](result_or_errbox<sp_obj> obj) mutable {
				res = obj ? obj.value()->type_id() : type_descriptor::nil().name;
				adbg(this) << "<- a_lnk_otid: " << res << std::endl;
			},
			ReqOpts::ErrorIfNOK | ReqOpts::DirectInvoke
		);
		return res;
	},

	// get node's group ID
	[this](a_node_gid) -> result_or_errbox<std::string> {
		adbg(this) << "<- a_node_gid" << std::endl;
		auto res = result_or_err<std::string>{};
		data_node_ex(
			[&](result_or_errbox<sp_node> N) {
				N.map([&](const sp_node& N) {
					res = N->gid();
				});
			},
			ReqOpts::ErrorIfNOK | ReqOpts::DirectInvoke
		);
		return res;
	},

	// obtain inode
	// [NOTE] assume it's a fast call, override behaviour where needed (sym_link for ex)
	[this](a_lnk_inode) -> result_or_errbox<inodeptr> {
		adbg(this) << "<- a_lnk_inode" << std::endl;
		return impl.get_inode();
	},

	[this](a_lnk_data, bool wait_if_busy) -> caf::result< result_or_errbox<sp_obj> > {
		adbg(this) << "<- a_lnk_data, status = " <<
			to_string(impl.status_[0].value) << "," << to_string(impl.status_[1].value) << std::endl;

		auto res = make_response_promise< result_or_errbox<sp_obj> >();
		data_ex(
			[=](result_or_errbox<sp_obj> obj) mutable { res.deliver(std::move(obj)); },
			(wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy) //| ReqOpts::Detached
		);
		return res;
	},

	// default handler for `data_node` that works via `data`
	[this](a_lnk_dnode, bool wait_if_busy) -> caf::result< result_or_errbox<sp_node> > {
		adbg(this) << "<- a_lnk_dnode, status = " <<
			to_string(impl.status_[0].value) << "," << to_string(impl.status_[1].value) << std::endl;

		auto res = make_response_promise< result_or_errbox<sp_node> >();
		data_node_ex(
			[=](result_or_errbox<sp_node> N) mutable { res.deliver(std::move(N)); },
			(wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy) //| ReqOpts::Detached
		);
		return res;
	},
}; }

/*-----------------------------------------------------------------------------
 *  fast_link_actor
 *-----------------------------------------------------------------------------*/
// for fast link we can assume that requests are invoked directly
// => override slow API to exclude extra delivery messages
auto fast_link_actor::make_behavior() -> behavior_type {
	return caf::message_handler({
		// obtain inode
		[this](a_lnk_inode) -> result_or_errbox<inodeptr> {
			adbg(this) << "<- a_lnk_inode" << std::endl;

			return impl.get_inode();
		},

		[this](a_lnk_data, bool) -> result_or_errbox<sp_obj> {
			adbg(this) << "<- a_lnk_data, status = " <<
				to_string(impl.status_[0].value) << "," << to_string(impl.status_[1].value) << std::endl;

			auto res = result_or_errbox<sp_obj>{};
			data_ex(
				[&](result_or_errbox<sp_obj> obj) { res = std::move(obj); },
				ReqOpts::WaitIfBusy
			);
			return res;
		},

		[this](a_lnk_dnode, bool) -> result_or_errbox<sp_node> {
			adbg(this) << "<- a_lnk_dnode, status = " <<
				to_string(impl.status_[0].value) << "," << to_string(impl.status_[1].value) << std::endl;

			auto res = result_or_errbox<sp_node>();
			data_node_ex(
				[&](result_or_errbox<sp_node> N) { res = std::move(N); },
				ReqOpts::WaitIfBusy
			);
			return res;
		},
	}).or_else(super::make_behavior());
}

/*-----------------------------------------------------------------------------
 *  other
 *-----------------------------------------------------------------------------*/
// extract timeout from kernel config
auto def_timeout(bool for_data) -> caf::duration {
	using namespace kernel::config;
	return caf::duration{ for_data ?
		get_or( config(), "radio.data-timeout", def_data_timeout ) :
		get_or( config(), "radio.timeout", timespan{100ms} )
	};
}

NAMESPACE_END(blue_sky::tree)
