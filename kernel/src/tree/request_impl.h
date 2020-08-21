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
#include <bs/detail/enumops.h>
#include "link_actor.h"
// [NOTE] Following includes are required (even though objects involved in messages
// are marked as unsafe CAF types) - for stringification inspector to work.
// Conclusion: all involved types to be formally serializable.
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

NAMESPACE_BEGIN(blue_sky::tree)

template<bool ManageStatus = true, typename F, typename C>
auto request_impl(
	link_actor& LA, Req req, ReqOpts opts, F f_request, C res_processor
) -> void {
	using namespace kernel::radio;
	using namespace allow_enumops;
	using f_ret_t = std::invoke_result_t<F>;
	using a_ret_t = result_or_errbox<typename f_ret_t::value_type>;
	using res_t = result_or_err<typename f_ret_t::value_type>;

	// if opts::Uniform is true, then both status values are changed at once
	// returns scalar prev state of `req` request
	auto rs_reset = [=](auto& LA, auto... xs) {
		if(enumval(opts) & enumval(ReqOpts::Uniform))
			LA.impl.rs_reset(req == Req::Data ? Req::DataNode : Req::Data, xs...);
		return LA.impl.rs_reset(req, xs...);
	};

	auto prev_rs = ReqStatus::OK;
	if constexpr(ManageStatus) {
		prev_rs = rs_reset(LA, ReqReset::IfNeq, ReqStatus::Busy, ReqStatus::OK);
		if(prev_rs == ReqStatus::OK) {
			if(enumval(opts & ReqOpts::HasDataCache)) opts &= ReqOpts::DirectInvoke;
		}
		else if(enumval(opts & ReqOpts::ErrorIfNOK)) {
			// restore status & deliver error
			rs_reset(LA, ReqReset::Always, prev_rs);
			res_processor(res_t{ unexpected_err_quiet(Error::EmptyData) });
			return;
		}
	}

	// returns extended request result processor
	const auto make_result = [&] {
		return [=, &LA, rs_reset = std::move(rs_reset), rp = std::move(res_processor)](a_ret_t obj) mutable {
			// set new status
			if constexpr(ManageStatus)
				rs_reset(
					LA, ReqReset::Always,
					obj ? (*obj ? ReqStatus::OK : ReqStatus::Void) : ReqStatus::Error, prev_rs,
					[&LA](Req req, ReqStatus new_rs, ReqStatus prev_rs) {
						link_impl::send_home<high_prio>(&LA, a_ack(), a_lnk_status(), req, new_rs, prev_rs);
					}
				);

			// if request result is nil -> return error
			const auto res = obj.and_then([](auto&& obj) {
				return obj ? res_t(std::move(obj)) : unexpected_err_quiet(Error::EmptyData);
			});
			// invoke main result processor
			rp(res);

			// feed waiters
			auto& waiters = LA.impl.req_status_handle(req).waiters;
			for(auto& w : waiters)
				(*reinterpret_cast<C*>(w()))(res);
			waiters.clear();
		};
	};

	// returns request result waiter
	const auto make_waiter = [&] {
		return [rp = std::move(res_processor)]() mutable -> void* { return &rp; };
	};
	// [NOTE] either `make_result()` or `make_waiter()` must be called once!

	// if we were in Busy state, then either deliver error or install waiter
	if(prev_rs == ReqStatus::Busy) {
		if(enumval(opts & ReqOpts::ErrorIfBusy))
			res_processor(res_t{ unexpected_err_quiet(Error::LinkBusy) });
		else
			LA.impl.req_status_handle(req).waiters.emplace_back(make_waiter());
	}
	// if specified, invoke request directly
	else if(enumval(opts & ReqOpts::DirectInvoke))
		make_result()(f_request());
	// otherwise invoke inside dedicated actor
	else {
		// setup worker that will invoke request and process result
		auto worker = [f = std::move(f_request)](caf::event_based_actor* self) mutable {
			return caf::behavior{
				[f = std::move(f)](a_apply) mutable -> a_ret_t { return f(); }
			};
		};

		// spawn worker
		auto worker_actor = enumval(opts & ReqOpts::Detached) ?
			system().spawn<caf::detached>(std::move(worker)) :
			system().spawn(std::move(worker));

		// start work & ensure that result_processor is invoked in LA's body
		LA.request(worker_actor, caf::infinite, a_apply())
		.then(make_result());
	}
}

template<bool ManageStatus = true, typename C>
auto data_node_request(link_actor& LA, ReqOpts opts, C res_processor) {
	using namespace allow_enumops;
	// make monlith request, as node is extracted from downloaded object
	request_impl<ManageStatus>(
		LA, Req::DataNode, opts | ReqOpts::Uniform,
		[Limpl = LA.pimpl_]() { return Limpl->data(); },
		[rp = std::move(res_processor)](const auto& maybe_obj) mutable {
			rp(maybe_obj.and_then( [&](const auto& obj) -> node_or_err {
				if(auto n = obj->data_node())
					return n;
				return unexpected_err_quiet(Error::NotANode);
			} ));
		}
	);
}

NAMESPACE_END(blue_sky::tree)
