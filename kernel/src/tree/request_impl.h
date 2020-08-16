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

template<bool ManageStatus = true, typename F, typename C>
auto request_impl(
	link_actor& LA, Req req, ReqOpts opts, F f_request, C res_processor
) -> void {
	using namespace kernel::radio;
	using namespace allow_enumops;
	using f_ret_t = std::invoke_result_t<F>;
	using a_ret_t = result_or_errbox<typename f_ret_t::value_type>;
	using ret_t = result_or_err<typename f_ret_t::value_type>;

	auto prev_rs = ReqStatus::OK;
	if constexpr(ManageStatus) {
		prev_rs = LA.impl.rs_reset(req, ReqReset::IfNeq, ReqStatus::Busy, ReqStatus::OK);
		if(prev_rs == ReqStatus::OK) {
			if(enumval(opts & ReqOpts::HasDataCache)) opts &= ReqOpts::DirectInvoke;
		}
		else if(enumval(opts & ReqOpts::ErrorIfNOK)) {
			// restore status & deliver error
			LA.impl.rs_reset(req, ReqReset::Always, prev_rs);
			res_processor(unexpected_err_quiet(Error::EmptyData));
			return;
		}
	}

	// returns extended request result processor
	const auto make_result = [&] {
		return [=, &LA, rp = std::move(res_processor)](a_ret_t obj) mutable {
			// set new status
			if constexpr(ManageStatus)
				LA.impl.rs_reset(
					req, ReqReset::Always,
					obj ? (*obj ? ReqStatus::OK : ReqStatus::Void) : ReqStatus::Error, prev_rs,
					[&LA](Req req, ReqStatus new_rs, ReqStatus prev_rs) {
						LA.send<high_prio>(LA.impl.home, a_ack(), a_lnk_status(), req, new_rs, prev_rs);
					}
				);

			// if result is nil -> return error
			auto res = obj.and_then([](auto&& obj) -> ret_t {
				return obj ? ret_t(std::move(obj)) : unexpected_err_quiet(Error::EmptyData);
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
			res_processor(unexpected_err_quiet(Error::LinkBusy));
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

	// [TODO] enable this code after simultaneous status manip is implemented
	//request_impl<ManageStatus>(
	//	LA, Req::DataNode, opts,
	//	[Limpl = LA.pimpl_]() { return Limpl->data(); },
	//	[rp = std::move(res_processor)](const auto& maybe_obj) mutable {
	//		rp(maybe_obj.and_then( [&](const auto& obj) -> node_or_err {
	//			if(auto n = obj->data_node())
	//				return n;
	//			return unexpected_err_quiet(Error::NotANode);
	//		} ));
	//	}
	//);

	request_impl<ManageStatus>(
		LA, Req::DataNode, opts,
		[&LA, opts]() mutable -> node_or_errbox {
			// directly invoke 'Data' request, store returned value in `res` and return it
			auto res = node_or_errbox{};
			request_impl<ManageStatus>(
				LA, Req::Data, static_cast<ReqOpts>(enumval(opts) | enumval(ReqOpts::DirectInvoke)),
				[&LA] {
					return LA.pimpl_->data().and_then([](const sp_obj& obj) -> node_or_err {
						if(obj) {
							if(auto n = obj->data_node())
								return n;
							return unexpected_err_quiet(Error::NotANode);
						}
						return unexpected_err_quiet(Error::EmptyData);
					});
				},
				[&res](node_or_errbox&& N) { res = std::move(N); }
			);
			return res;
		},
		std::move(res_processor)
	);
}

NAMESPACE_END(blue_sky::tree)
