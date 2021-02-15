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

[[maybe_unused]] auto adbg_impl(caf::actor_ostream out, const link_impl& L) -> caf::actor_ostream {
	out << "[L:" << L.type_id() << "] [" << to_string(L.id_) << "] [" << L.name_ << "]: ";
	return out;
}

// helper to apply data transactions
template<typename... Ts>
static auto do_data_apply(link_actor* LA, transaction_t<tr_result, Ts...> tr) {
	auto res = LA->make_response_promise<tr_result::box>();
	LA->request(LA->actor(), caf::infinite, a_data(), true)
	.then([=, tr = std::move(tr)](obj_or_errbox maybe_obj) mutable {
		if(maybe_obj) {
			// always send closed transaction
			res.delegate((*maybe_obj)->actor(), a_apply(), [&] {
				if constexpr(sizeof...(Ts) > 0)
					return (*maybe_obj)->make_transaction(std::move(tr));
				else
					return std::move(tr);
			}());
		}
		else
			res.deliver(pack(tr_result{std::move(maybe_obj).error()}));
	});
	return res;
}

/*-----------------------------------------------------------------------------
 *  link
 *-----------------------------------------------------------------------------*/
link_actor::link_actor(caf::actor_config& cfg, caf::group home, sp_limpl Limpl) :
	super(cfg, std::move(home), std::move(Limpl)), ropts_{ReqOpts::WaitIfBusy, ReqOpts::WaitIfBusy}
{}

auto link_actor::name() const -> const char* {
	return "link_actor";
}

///////////////////////////////////////////////////////////////////////////////
//  behavior
//
auto link_actor::make_primary_behavior() -> primary_actor_type::behavior_type {
return {

	[=](a_home) { return impl.home; },

	[=](a_home_id) { return std::string(impl.home_id()); },

	[=](a_impl) -> sp_limpl { return spimpl(); },

	[=](a_clone, a_impl, bool deep) { return impl.clone(this, deep); },

	[=](a_clone, bool deep) {
		auto res = make_response_promise<link>();
		request(actor(), caf::infinite, a_clone{}, a_impl{}, deep)
		.then([=](sp_limpl Limpl) mutable {
			res.deliver( link(std::move(Limpl)) );
		});
		return res;
	},

	// subscribe events listener
	[=](a_subscribe, const caf::actor& baby) {
		// remember baby & ensure it's alive
		return delegate(caf::actor_cast<ev_listener_actor_type>(baby), a_hi(), impl.home);
	},

	// apply link transaction
	[=](a_apply, const simple_transaction& tr) -> error::box {
		adbg(this) << "<- a_apply simple_transaction" << std::endl;
		return tr_eval(tr);
	},

	[=](a_apply, const link_transaction& tr) -> error::box {
		return tr_eval(tr, bare_link(spimpl()));
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
	[=](a_lnk_status, Req req) -> ReqStatus { return impl.req_status(req); },

	// change status
	[=](a_lnk_status, Req req, ReqReset cond, ReqStatus new_rs, ReqStatus prev_rs) -> ReqStatus {
		adbg(this) << "<- a_lnk_status: " << to_string(req) << " " <<
			to_string(prev_rs) << "->" << to_string(new_rs) << std::endl;
		return impl.rs_reset(req, cond, new_rs, prev_rs);
	},

	// just send notification about just changed status
	[=](a_lnk_status, Req req, ReqStatus new_rs, ReqStatus prev_rs) {
		impl.send_home<high_prio>(this, a_ack(), a_lnk_status(), req, new_rs, prev_rs);
	},

	// get/set flags
	[=](a_lnk_flags) { return impl.flags_; },
	[=](a_lnk_flags, Flags f) { impl.flags_ = f; },

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

		return request_data(
			*this, ropts_.data | (wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy)
		);
	},

	// get data node
	[=](a_data_node, bool wait_if_busy) -> caf::result< node_or_errbox > {
		adbg(this) << "<- a_data_node, status = " <<
			to_string(impl.req_status(Req::Data)) << "," << to_string(impl.req_status(Req::DataNode)) << std::endl;

		return request_data_node(
			*this, ropts_.data_node | (wait_if_busy ? ReqOpts::WaitIfBusy : ReqOpts::ErrorIfBusy)
		);
	},

	// noop - implement in derived links
	[=](a_lazy, a_load, bool) { return false; }
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
		// if `with_node` is true, then do the same for DataNode request
		[=](a_lazy, a_load, bool with_node) {
			auto orig_me = current_behavior();

			// setup request impl that invokes `a_delay_load` on object once
			const auto load_then_answer = [&](auto req) {
				using req_t = decltype(req);
				using R = std::conditional_t<std::is_same_v<req_t, a_data>, obj_or_errbox, node_or_errbox>;
				constexpr auto req_id = [&] {
					if constexpr(std::is_same_v<req_t, a_data>) return Req::Data;
					else return Req::DataNode;
				}();

				return [=](req_t, bool) mutable -> caf::result<R> {
					// this handler triggered only once
					become(orig_me);

					impl.rs_reset(req_id, ReqReset::Always, ReqStatus::Busy);
					// drop lazy load flag
					impl.flags_ &= ~Flags::LazyLoad;
					// get cached object
					auto obj = impl.data(unsafe);
					if(!obj) return unexpected_err_quiet(Error::EmptyData);

					auto res = make_response_promise<R>();
					request(objbase_actor::actor(*obj), caf::infinite, a_load(), std::move(obj))
					.then(
						[=](error::box er) mutable {
							// if error happened - deliver it
							// [NOTE] assume that req status = status of lazy load op
							if(er.ec) {
								impl.rs_reset(req_id, ReqReset::Always, ReqStatus::Error);
								res.deliver(R{ tl::unexpect, std::move(er) });
							}
							// otherwise, forward data request to self (with orig_me installed)
							else {
								impl.rs_reset(req_id, ReqReset::Always, ReqStatus::OK);
								res.delegate(actor(this), req_t(), true);
							}
						},

						[=](const caf::error& er) mutable {
							impl.rs_reset(req_id, ReqReset::Always, ReqStatus::Error);
							res.deliver(R{ tl::unexpect, forward_caf_error(er) });
						}
					);
					return res;
				};
			};

			auto lazy_me = caf::message_handler{ load_then_answer(a_data()) };
			if(with_node) {
				// raise lazy load flag
				impl.flags_ |= Flags::LazyLoad;
				lazy_me = lazy_me.or_else( load_then_answer(a_data_node()) );
			}
			// setup new behavior for Data & DataNode requests
			become(lazy_me.or_else(orig_me));
			return true;
		}
	}, super::make_typed_behavior() );
}

auto cached_link_actor::make_behavior() -> behavior_type {
	return make_typed_behavior().unbox();
}

NAMESPACE_END(blue_sky::tree)
