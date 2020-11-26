/// @author uentity
/// @date 25.11.2019
/// @brief Implements generic frame for making link data requests
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "link_actor.h"

#include <bs/detail/enumops.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

NAMESPACE_BEGIN(blue_sky::tree)
NAMESPACE_BEGIN(detail)

template<typename R, typename F>
struct request_traits {
	// helper that passes pointer to bg actor making a request if functor needs it
	static constexpr auto invoke_f_request(F& f, caf::event_based_actor* origin) {
		if constexpr(std::is_invocable_v<F, caf::event_based_actor*>)
			return f(origin);
		else
			return f();
	};

	using f_ret_t = std::invoke_result_t<decltype(invoke_f_request), F&, caf::event_based_actor*>;
	static constexpr bool can_invoke_inplace = tl::detail::is_expected<f_ret_t>::value;

	// deduce request result type (must be result_or_errbox<R>)
	using R_deduced = std::conditional_t<std::is_same_v<R, void>, f_ret_t, result_or_errbox<R>>;
	static_assert(tl::detail::is_expected<R_deduced>::value);
	// assume request result is `result_or_errbox<R>`
	using res_t = result_or_errbox<typename R_deduced::value_type>;
	// value type that worker actor returns (always transfers error in a box)
	using worker_ret_t = std::conditional_t<can_invoke_inplace, res_t, f_ret_t>;
};

NAMESPACE_END(detail)

// `f_request` must return `result_or_errbox<R>`, `R` can be specified explicitly or auto-deduced.
// Otherwise matching is not guranteed if work is started in standalone actor.
// If opts::Uniform is true, then both status values are changed at once.
// Returns worker actor handle, result can be obtain in lazy manner by sending `a_ack` message to it
template<typename R = void, typename F>
auto request_impl(link_actor& LA, Req req, ReqOpts opts, F f_request) -> result_or_err<caf::actor> {
	using namespace kernel::radio;
	using namespace allow_enumops;

	using rtraits = detail::request_traits<R, F>;
	using res_t = typename rtraits::res_t;
	using worker_ret_t = typename rtraits::worker_ret_t;

	auto err_if_nok = false;
	const auto req_transaction = [&](Req req, ReqStatus new_rs, ReqStatus old_rs) {
		// restore status if we were only allowed to run in OK state
		if(enumval(opts & ReqOpts::ErrorIfNOK)) {
			err_if_nok = true;
			return false;
		}
		else {
			// send notification
			link_impl::send_home<high_prio>(&LA, a_ack(), a_lnk_status(), req, new_rs, old_rs);
			return true;
		}
	};

	// change status to Busy if it wasn't OK
	auto cond = ReqReset::IfNeq;
	if(enumval(opts) & enumval(ReqOpts::Uniform)) cond |= ReqReset::Broadcast;
	const auto prev_rs = LA.impl.rs_reset(req, cond, ReqStatus::Busy, ReqStatus::OK, req_transaction);

	// early error reporting
	if(err_if_nok)
		return unexpected_err_quiet(Error::EmptyData);
	if(prev_rs == ReqStatus::Busy && enumval(opts & (ReqOpts::ErrorIfBusy | ReqOpts::DirectInvoke)))
		return unexpected_err_quiet(Error::LinkBusy);

	// detect inplace invoke
	if constexpr(rtraits::can_invoke_inplace) {
		if(prev_rs == ReqStatus::OK && enumval(opts & ReqOpts::HasDataCache))
			opts |= ReqOpts::DirectInvoke;
	}
	// return OK error to indicate that work mst be invoked inplace
	if(enumval(opts & ReqOpts::DirectInvoke))
		return unexpected_err_quiet(perfect);

	// don't start detached actor if status is OK or we start waiting
	if(prev_rs == ReqStatus::OK || prev_rs == ReqStatus::Busy)
		opts &= ~ReqOpts::Detached;

	// worker stateful actor that will invoke request and process result
	// we need state to share lazily response promise
	struct rstate {
		std::optional<caf::typed_response_promise<res_t>> res;

		constexpr decltype(auto) get(caf::event_based_actor* self) {
			if(!res) res = self->make_response_promise<res_t>();
			return *res;
		}
	};

	// worker stateful actor that will invoke request and process result or wait for it
	auto worker = [=, f = std::move(f_request), Limpl = LA.spimpl()]
	(caf::stateful_actor<rstate>* self) mutable -> caf::behavior {
		// if we were in busy state, install self as waiter
		if(prev_rs == ReqStatus::Busy) {
			// prepare request result processor
			auto rp = [self](res_t obj) mutable {
				// deliver result to waiting client & quit
				self->state.get(self).deliver(std::move(obj));
				self->quit();
			};

			// fallback CAF errors handler
			self->set_error_handler(
				[rp](auto*, const caf::error& er) mutable {
					rp(res_t{ tl::unexpect, forward_caf_error(er) });
				}
			);

			// add self to waiting queue
			Limpl->rs_add_waiter(req, caf::actor_cast<caf::actor>(self));
			// install behavior
			return {
				// deliver result from waiting state
				[rp = std::move(rp)](a_apply, res_t res) mutable {
					rp(std::move(res));
				},
				// `a_ack` simply returns result promise
				[self](a_ack) { return self->state.get(self); }
			};
		}

		// ... otherwise setup for performing request
		// prepare request result processor
		auto rp = [self, Limpl = std::move(Limpl)](res_t obj) mutable {
			if constexpr(!rtraits::can_invoke_inplace) {
				// deliver result to waiting client
				self->state.get(self).deliver(obj);
			}
			// update status & quit
			Limpl->rs_update_from_data(std::move(obj));
			self->quit();
		};

		// fallback CAF errors handler
		self->set_error_handler(
			[rp](auto*, const caf::error& er) mutable {
				rp(res_t{ tl::unexpect, forward_caf_error(er) });
			}
		);

		// install behavior
		if constexpr(rtraits::can_invoke_inplace) {
			return {
				// do complete processing on `a_ack` message
				[self, rp = std::move(rp), f = std::move(f)](a_ack) mutable {
					res_t res = rtraits::invoke_f_request(f, self);
					// update status inplace
					rp(res);
					return res;
				}
			};
		}
		else {
			return {
				// invoke request functor & return result
				[self, f = std::move(f)](a_apply) mutable -> worker_ret_t {
					return rtraits::invoke_f_request(f, self);
				},
				// launch request on `a_ack` message
				[self, rp = std::move(rp)](a_ack) mutable -> caf::result<res_t> {
					// launch work
					auto res_promise = self->state.get(self);
					self->request(caf::actor_cast<caf::actor>(self), caf::infinite, a_apply())
					.then(std::move(rp));
					return res_promise;
				}
			};
		}
	};

	// spawn & return worker
	return enumval(opts & ReqOpts::Detached) ?
		LA.spawn<caf::detached>(std::move(worker)) :
		LA.spawn(std::move(worker));
}

// non-lazy version that immediately starts processing
template<typename R = void, typename F, typename C>
auto request_impl(
	link_actor& LA, Req req, ReqOpts opts, F f_request, C res_processor
) -> void {
	using namespace allow_enumops;
	using rtraits = detail::request_traits<R, F>;
	using res_t = typename rtraits::res_t;

	request_impl<R>(LA, req, opts, f_request)
	.map([&](auto&& req_worker) {
		LA.request(req_worker, caf::infinite, a_ack()).then(
			[=](res_t r) mutable { res_processor(std::move(r)); },
			[=](const caf::error& er) mutable {
				res_processor(res_t{ tl::unexpect, forward_caf_error(er) });
			}
		);
	})
	.or_else([&](const auto& er) {
		if(er.ok()) {
			if constexpr(rtraits::can_invoke_inplace) {
				// invoke request inplace & update status
				auto res = rtraits::invoke_f_request(f_request, &LA);
				LA.impl.rs_update_from_data(res);
				// invoke extra processor
				res_processor(std::move(res));
			}
			else
				res_processor(unexpected_err("Can't invoke request inplace"));
		}
		else
			res_processor(res_t{ tl::unexpect, er });
	});
}

template<typename C>
auto request_data(link_actor& LA, ReqOpts opts, C res_processor) {
	request_impl(
		LA, Req::Data, opts,
		[Limpl = LA.spimpl()] { return Limpl->data(); },
		std::move(res_processor)
	);
}

template<typename C>
auto request_data(unsafe_t, link_actor& LA, ReqOpts opts, C res_processor) {
	request_impl(
		LA, Req::Data, opts,
		[Limpl = LA.spimpl()]() -> obj_or_errbox { return Limpl->data(unsafe); },
		std::move(res_processor)
	);
}

template<typename C>
auto request_data_node(link_actor& LA, ReqOpts opts, C res_processor) {
	using namespace allow_enumops;
	// make uniform request, as node is extracted from downloaded object
	request_impl(
		LA, Req::DataNode, opts | ReqOpts::Uniform,
		[Limpl = LA.spimpl()]() { return Limpl->data(); },
		[rp = std::move(res_processor)](const auto& maybe_obj) mutable {
			rp(maybe_obj.and_then( [&](const auto& obj) -> node_or_err {
				if(auto n = obj->data_node())
					return n;
				return unexpected_err_quiet(Error::NotANode);
			} ));
		}
	);
}

template<typename C>
auto request_data_node(unsafe_t, link_actor& LA, ReqOpts opts, C res_processor) {
	request_impl(
		LA, Req::DataNode, opts,
		[Limpl = LA.spimpl()]() -> node_or_errbox { return Limpl->data_node(unsafe); },
		std::move(res_processor)
	);
}

NAMESPACE_END(blue_sky::tree)
