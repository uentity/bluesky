/// @file
/// @author uentity
/// @date 25.11.2019
/// @brief Implements generic frame for making link data requests
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/kernel/radio.h>
#include "link_actor.h"
// [NOTE] Following includes are required (even though objects involved in messages
// are marked as unsafe CAF types) - for stringification inspector to work.
// Conclusion: all involved types to be formally serializable.
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

NAMESPACE_BEGIN(blue_sky::tree)
NAMESPACE_BEGIN()

template<bool ManageStatus = true, typename F, typename C>
auto request_impl(
	link_actor& LA, Req req, ReqOpts opts, F&& f_request, C&& res_processor
) -> void {
	using namespace kernel::radio;
	using namespace allow_enumops;
	using f_ret_t = std::invoke_result_t<F>;
	using a_ret_t = result_or_errbox<typename f_ret_t::value_type>;
	using ret_t = result_or_err<typename f_ret_t::value_type>;

	auto prev_rs = ReqStatus::Void;
	if constexpr(ManageStatus) {
		prev_rs = LA.impl.rs_reset(req, ReqReset::IfNeq, ReqStatus::Busy, ReqStatus::OK);
		if(prev_rs == ReqStatus::OK && enumval(opts & ReqOpts::HasDataCache))
			opts &= ReqOpts::DirectInvoke;
	}

	// setup request result post-processor
	auto make_result = [&LA, rp = std::forward<C>(res_processor), req, prev_rs](a_ret_t obj) mutable {
		// set new status
		if constexpr(ManageStatus)
			LA.impl.rs_reset(
				req, ReqReset::Always,
				obj ? (obj.value() ? ReqStatus::OK : ReqStatus::Void) : ReqStatus::Error, prev_rs,
				[&LA](Req req, ReqStatus new_rs, ReqStatus prev_rs) {
					LA.send<high_prio>(LA.impl.home, a_ack(), a_lnk_status(), req, new_rs, prev_rs);
				}
			);

		// invoke result processor
		std::invoke(
			std::forward<C>(rp),
			obj.and_then([](auto&& obj) -> ret_t {
				return obj ?
					ret_t(std::move(obj)) :
					tl::make_unexpected(error::quiet(Error::EmptyData));
			})
		);
	};

	// check starting conditions
	if constexpr(ManageStatus) {
		if(prev_rs != ReqStatus::OK && enumval(opts & ReqOpts::ErrorIfNOK)) {
			make_result( tl::make_unexpected(error::quiet(Error::EmptyData)) );
			return;
		}
		if(prev_rs == ReqStatus::Busy && enumval(opts & ReqOpts::ErrorIfBusy)) {
			make_result( tl::make_unexpected(error::quiet(Error::LinkBusy)) );
			return;
		}
	}

	// invoke request directly or inside spawned actor
	if(enumval(opts & ReqOpts::DirectInvoke)) {
		make_result( std::invoke(std::forward<F>(f_request)) );
	}
	else {
		// setup worker that will actually invoke `impl.data()`
		auto worker = [f = std::forward<F>(f_request)](caf::event_based_actor* self) mutable
		-> caf::behavior {
			return {
				[f = std::forward<F>(f)](a_ack) mutable -> a_ret_t { return std::invoke(std::forward<F>(f)); }
			};
		};

		// spawn worker
		auto worker_actor = enumval(opts & ReqOpts::Detached) ?
			system().spawn<caf::detached>(std::move(worker)) :
			system().spawn(std::move(worker));

		// make request and invoke result processor
		LA.request(worker_actor, def_timeout(true), a_ack())
		.then(std::move(make_result));
	}
}

NAMESPACE_END()
NAMESPACE_END(blue_sky::tree)
