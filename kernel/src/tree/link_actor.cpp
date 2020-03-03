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
#include "actor_debug.h"

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::data_modificator_f)

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;
using namespace std::chrono_literals;
using bs_detail::shared;

#if DEBUG_ACTOR == 1

static auto adbg(link_actor* A) -> caf::actor_ostream {
	return caf::aout(A) << "[L] [" << to_string(A->impl.id_) <<
		"] [" << A->impl.name_ << "]: ";
}

#endif

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
	adbg(this) << "joined self group " << impl.home.get()->identifier() << std::endl;

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

auto link_actor::on_exit() -> void {
	adbg(this) << "dies" << std::endl;

	// be polite with everyone
	goodbye();
	// force release strong ref to link's impl
	pimpl_.reset();
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
	adbg(this) << "<- a_lnk_rename " << (silent ? "silent: " : "loud: ") << impl.name_ <<
		" -> " << new_name << std::endl;

	auto old_name = impl.name_;
	impl.name_ = std::move(new_name);
	// send rename ack message
	if(!silent)
		send<high_prio>(impl.home, a_ack(), a_lnk_rename(), impl.name_, std::move(old_name));
}


///////////////////////////////////////////////////////////////////////////////
//  behavior
//
auto link_actor::make_typed_behavior() -> typed_behavior {
	return typed_behavior{
		// skip `bye` message (should always come from myself)
		[=](a_bye) -> void {
			adbg(this) << "<- a_lnk_bye " << std::endl;
		},

		/// 1. get id
		[=](a_lnk_id) -> lid_type {
			adbg(this) << "<- a_lnk_id: " << to_string(impl.id_) << std::endl;
			return impl.id_;
		},

		// 2. get oid
		[=](a_lnk_oid) -> std::string {
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

		// 3. get object type_id
		[=](a_lnk_otid) -> std::string {
			// [NOTE] assume that if status is OK then getting data is fast (data is cached)
			auto res = std::string{};
			//auto tstart = make_timestamp();
			data_ex(
				[&](result_or_errbox<sp_obj> obj) mutable {
					res = obj ? obj.value()->type_id() : nil_otid;
					adbg(this) << "<- a_lnk_otid: " << res << std::endl;
				},
				ReqOpts::ErrorIfNOK | ReqOpts::DirectInvoke
			);
			return res;
		},

		// 4. get node's group ID
		[=](a_node_gid) -> result_or_errbox<std::string> {
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

		/// 5. get name
		[=](a_lnk_name) -> std::string {
			adbg(this) << "<- a_lnk_name: " << impl.name_ << std::endl;
			return impl.name_;
		},

		/// 6. rename
		[=](a_lnk_rename, std::string new_name, bool silent) -> void {
			rename(std::move(new_name), silent);
		},
		// 7. rename ack
		[=](a_ack, a_lnk_rename, std::string new_name, const std::string& old_name) -> void {
			adbg(this) << "<- a_lnk_rename ack: " << old_name << "->" << new_name << std::endl;
			if(current_sender() != this)
				rename(std::move(new_name), true);
		},

		// 8. get status
		[=](a_lnk_status, Req req) -> ReqStatus { return pimpl_->req_status(req); },

		// 9. change status
		[=](a_lnk_status, Req req, ReqReset cond, ReqStatus new_rs, ReqStatus prev_rs) -> ReqStatus {
			adbg(this) << "<- a_lnk_status: " << to_string(req) << " " <<
				to_string(prev_rs) << "->" << to_string(new_rs) << std::endl;
			return impl.rs_reset(
				req, cond, new_rs, prev_rs,
				[=](Req req, ReqStatus new_s, ReqStatus old_s) {
					send<high_prio>(impl.home, a_ack(), a_lnk_status(), req, new_s, old_s);
				}
			);
		},

		// 10. reset status ack
		[=](a_ack, a_lnk_status, Req req, ReqStatus new_s, ReqStatus prev_s) {
			adbg(this) << "<- a_lnk_status ack: " << to_string(req) << " " <<
				to_string(prev_s) << "->" << to_string(new_s) << std::endl;
		},

		// 11, 12. get/set flags
		[=](a_lnk_flags) { return pimpl_->flags_; },
		[=](a_lnk_flags, Flags f) { pimpl_->flags_ = f; },

		// 13. obtain inode
		// [NOTE] assume it's a fast call, override behaviour where needed (sym_link for ex)
		[=](a_lnk_inode) -> result_or_errbox<inodeptr> {
			adbg(this) << "<- a_lnk_inode" << std::endl;
			return impl.get_inode();
		},

		// 14. get data
		[=](a_lnk_data, bool wait_if_busy) -> caf::result< result_or_errbox<sp_obj> > {
			adbg(this) << "<- a_lnk_data, status = " <<
				to_string(impl.status_[0].value) << "," << to_string(impl.status_[1].value) << std::endl;

			auto res = make_response_promise< result_or_errbox<sp_obj> >();
			data_ex(
				[=](result_or_errbox<sp_obj> obj) mutable { res.deliver(std::move(obj)); },
				(wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy) //| ReqOpts::Detached
			);
			return res;
		},

		// 15. get data node
		[=](a_lnk_dnode, bool wait_if_busy) -> caf::result< result_or_errbox<sp_node> > {
			adbg(this) << "<- a_lnk_dnode, status = " <<
				to_string(impl.status_[0].value) << "," << to_string(impl.status_[1].value) << std::endl;

			auto res = make_response_promise< result_or_errbox<sp_node> >();
			data_node_ex(
				[=](result_or_errbox<sp_node> N) mutable { res.deliver(std::move(N)); },
				(wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy) //| ReqOpts::Detached
			);
			return res;
		},

		// 16. apply modifier function on pointee
		// set `Data` status depending on error returned from modifier
		[=](a_apply, data_modificator_f m, bool silent) mutable -> caf::result<error::box> {
			auto res = make_response_promise();

			request(caf::actor{this}, caf::duration{def_timeout(true)}, a_lnk_data(), true)
			.then([=, m = std::move(m)](result_or_errbox<sp_obj> obj) mutable {
				// invoke modificator
				auto er = obj ?
					error::eval_safe( [&]{ return m(std::move(obj.value())); } ) :
					error::unpack(obj.error());
				// set status
				pimpl_->rs_reset(
					Req::Data, ReqReset::Always, er.ok() ? ReqStatus::OK : ReqStatus::Error, ReqStatus::Void,
					silent ?
						function_view{ link_impl::on_rs_changed_noop } :
						[=](Req req, ReqStatus new_s, ReqStatus old_s) {
							send<high_prio>(impl.home, a_ack(), a_lnk_status(), req, new_s, old_s);
						}
				);
				// deliver error back to callee
				res.deliver(er.pack());
			});
			return const_cast<const caf::response_promise&>(res);
		}
	};
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

		[=](a_lnk_data, bool) -> result_or_errbox<sp_obj> {
			adbg(this) << "<- a_lnk_data fast, status = " <<
				to_string(impl.status_[0].value) << "," << to_string(impl.status_[1].value) << std::endl;

			return pimpl_->data().and_then([](auto&& obj) {
				return obj ?
					result_or_errbox<sp_obj>{ std::move(obj) } :
					tl::make_unexpected(error::quiet(Error::EmptyData));
			});

			//auto res = result_or_errbox<sp_obj>{};
			//data_ex(
			//	[&](result_or_errbox<sp_obj> obj) { res = std::move(obj); },
			//	ReqOpts::WaitIfBusy
			//);
			//return res;
		},

		[=](a_lnk_dnode, bool) -> result_or_errbox<sp_node> {
			adbg(this) << "<- a_lnk_dnode fast, status = " <<
				to_string(impl.status_[0].value) << "," << to_string(impl.status_[1].value) << std::endl;

			return pimpl_->data().and_then([](auto&& obj) {
				return obj ?
					( obj->is_node() ?
						result_or_err<sp_node>(std::static_pointer_cast<tree::node>(std::move(obj))) :
						tl::make_unexpected(error::quiet(Error::NotANode))
					) :
					tl::make_unexpected(error::quiet(Error::EmptyData));
			});

			//auto res = result_or_errbox<sp_node>();
			//data_node_ex(
			//	[&](result_or_errbox<sp_node> N) { res = std::move(N); },
			//	ReqOpts::WaitIfBusy
			//);
			//return res;
		},

		[=](a_lnk_dcache) -> sp_obj {
			return static_cast<hard_link_impl&>(impl).data_;
		}
	}, super::make_typed_behavior() );
}

auto fast_link_actor::make_behavior() -> behavior_type {
	return make_typed_behavior().unbox();
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
