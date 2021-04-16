/// @author uentity
/// @date 25.11.2019
/// @brief Implements generic frame for making link data requests
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "link_actor.h"
#include "../kernel/radio_subsyst.h"

#include <bs/detail/enumops.h>

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

	using f_ret_t = decltype( invoke_f_request(std::declval<F&>(), std::declval<caf::event_based_actor*>()) );
	static constexpr bool can_invoke_inplace = tl::detail::is_expected<f_ret_t>::value;

	// deduce request result type (must be result_or_errbox<R>)
	using R_deduced = std::conditional_t<std::is_same_v<R, void>, f_ret_t, result_or_errbox<R>>;
	static_assert(tl::detail::is_expected<R_deduced>::value);
	// assume request result is `result_or_errbox<R>`
	using res_t = result_or_errbox<typename R_deduced::value_type>;
	// value type that worker actor returns (always transfers error in a box)
	using worker_ret_t = std::conditional_t<can_invoke_inplace, res_t, f_ret_t>;
	// type of extra result processing functor
	using custom_rp_f = std::function< void(res_t) >;
	// worker actor iface
	using actor_type = caf::typed_actor<
		typename caf::replies_to<a_ack>::template with<res_t>,
		typename caf::replies_to<a_ack, custom_rp_f>::template with<res_t>
	>;

	template<typename U, typename V>
	static auto chain_rp(U base_rp, V extra_rp) {
		return [base_rp = std::move(base_rp), extra_rp = std::move(extra_rp)](res_t obj) mutable {
			error::eval_safe([&] { base_rp(obj); });
			if constexpr(!std::is_same_v<V, std::nullopt_t>)
				error::eval_safe([&] { extra_rp(std::move(obj)); });
		};
	};
};

// `f_request` must return `result_or_errbox<R>`, `R` can be specified explicitly or auto-deduced.
// Otherwise matching is not guranteed if work is started in standalone actor.
// If opts::Uniform is true, then both status values are changed at once.
// Returns worker actor handle, result can be obtain in lazy manner by sending `a_ack` message to it
template<typename R = void, typename F>
auto make_request_actor(ReqStatus prev_rs, link_actor& LA, Req req, ReqOpts opts, F f_request) -> caf::actor {
	using rtraits = detail::request_traits<R, F>;
	using res_t = typename rtraits::res_t;
	using worker_ret_t = typename rtraits::worker_ret_t;

	// we need state to share response promise
	struct rstate {
		std::optional<caf::typed_response_promise<res_t>> res;

		constexpr decltype(auto) get(caf::event_based_actor* self) {
			if(!res) res = self->make_response_promise<res_t>();
			return *res;
		}
	};

	// define stateful actor that will invoke request and process result or wait for it
	auto worker = [=, f = std::move(f_request), Limpl = LA.spimpl()]
	(caf::stateful_actor<rstate>* self) mutable -> caf::behavior {
		using custom_rp_f = typename rtraits::custom_rp_f;

		// release worker from kernel if being tracked
		if(enumval(opts & ReqOpts::TrackWorkers))
			self->attach_functor([=] { KRADIO.release_citizen(self); });

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

			// install behavior
			return {
				// invoke result processor on delivered result
				[rp](a_apply, res_t res) mutable { rp(std::move(res)); },

				// `a_ack` simply returns result promise
				[self](a_ack) { return self->state.get(self); },

				// invoke additional result processor on `a_apply` message
				[self, rp](a_ack, custom_rp_f extra_rp) mutable {
					self->become(caf::message_handler{
						[rp = rtraits::chain_rp(std::move(rp), std::move(extra_rp))](a_apply, res_t res)
						mutable {
							rp(std::move(res));
						}
					}.or_else(self->current_behavior()));

					return self->state.get(self);
				}
			};
		}

		// ... otherwise perform request on `a_ack` message
		// prepare request result processor
		auto rp = [self, opts, Limpl = std::move(Limpl)](res_t obj) mutable {
			// update status
			Limpl->rs_update_from_data(obj, enumval(opts & ReqOpts::Uniform));
			// deliver result to waiting client
			if constexpr(!rtraits::can_invoke_inplace)
				self->state.get(self).deliver(std::move(obj));
			// and quit
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
			// helper to generate handler for `a_ack` message
			auto handle_ack = [self, rp = std::move(rp), f = std::move(f)](auto extra_rp) mutable {
				res_t res = rtraits::invoke_f_request(f, self);
				// update status inplace
				rtraits::chain_rp(rp, std::move(extra_rp))(res);
				return res;
			};

			return {
				// do complete processing on `a_ack` message
				[handle_ack](a_ack) mutable {
					return handle_ack(std::nullopt);
				},
				// same with extra result processing
				[handle_ack](a_ack, custom_rp_f extra_rp) mutable {
					return handle_ack(std::move(extra_rp));
				}
			};
		}
		else {
			// helper to generate handler for `a_ack` message
			auto handle_ack = [self, rp = std::move(rp)](auto extra_rp) mutable -> caf::result<res_t> {
				// launch work
				self->request(caf::actor_cast<caf::actor>(self), caf::infinite, a_apply())
				.then(rtraits::chain_rp(rp, std::move(extra_rp)));
				// return result promise
				return self->state.get(self);
			};

			return {
				// invoke request functor & return result
				[self, f = std::move(f)](a_apply) mutable -> worker_ret_t {
					return rtraits::invoke_f_request(f, self);
				},
				// launch request on `a_ack` message
				[handle_ack](a_ack) mutable {
					return handle_ack(std::nullopt);
				},
				// request with extra res processor
				[handle_ack](a_ack, custom_rp_f extra_rp) mutable {
					return handle_ack(std::move(extra_rp));
				}
			};
		}
	};

	// spawn worker actor
	auto res = enumval(opts & ReqOpts::Detached) ?
		LA.spawn<caf::detached>(std::move(worker)) :
		LA.spawn(std::move(worker));
	// early register worker if required
	if(enumval(opts & ReqOpts::TrackWorkers))
		KRADIO.register_citizen(res->address());
	return res;
}

NAMESPACE_END(detail)

// Launch request actor or return error to indicate fail or inplace invoke (error == OK)
template<typename R = void, typename F>
auto request_impl(link_actor& LA, Req req, ReqOpts opts, F f_request)
-> result_or_err<typename detail::request_traits<R, F>::actor_type> {
	using rtraits = detail::request_traits<R, F>;
	using res_t = result_or_err<typename rtraits::actor_type>;
	auto res = std::optional<res_t>{};

	const auto request_tr = [&](Req req, ReqStatus new_rs, ReqStatus prev_rs, link_impl::status_handle& S) {
		// check if unable to perform request
		if(enumval(opts & ReqOpts::ErrorIfNOK) && prev_rs != ReqStatus::OK)
			res.emplace(unexpected_err_quiet(Error::EmptyData));
		else if(prev_rs == ReqStatus::Busy && enumval(opts & (ReqOpts::ErrorIfBusy | ReqOpts::DirectInvoke)))
			res.emplace(unexpected_err_quiet(Error::LinkBusy));
		// revert status (return false) in case of error
		if(res) return false;

		// by default, change to Busy state only if was in non-OK
		bool do_change_status = prev_rs != ReqStatus::OK && prev_rs != ReqStatus::Busy;

		// if possible, invoke req inplace if status was OK
		if constexpr(rtraits::can_invoke_inplace) {
			if(prev_rs == ReqStatus::OK && enumval(opts & ReqOpts::HasDataCache))
				opts |= ReqOpts::DirectInvoke;
		}

		// if request must be ivoked inplace return OK error, otherwise spawn worker actor
		if(enumval(opts & ReqOpts::DirectInvoke)) {
			res.emplace(unexpected_err_quiet(perfect));
		}
		else {
			// don't start detached actor if status is OK or we start waiting
			if(prev_rs == ReqStatus::OK || prev_rs == ReqStatus::Busy)
				opts &= ~ReqOpts::Detached;

			// spawn request worker actor
			auto rworker = detail::make_request_actor<R>(prev_rs, LA, req, opts, std::move(f_request));
			// add to waiting queue if needed
			if(prev_rs == ReqStatus::Busy)
				S.waiters.push_back(rworker);

			res = caf::actor_cast<typename rtraits::actor_type>(std::move(rworker));
			// if we do some non-waiting work, we must change to Busy
			do_change_status = prev_rs != ReqStatus::Busy;
		}

		// change to Busy state only if was in non-OK
		if(do_change_status) {
			// send notification
			link_impl::send_home<high_prio>(&LA, a_ack(), a_lnk_status(), req, new_rs, prev_rs);
			return true;
		}
		return false;
	};

	// change status to Busy & invoke request transaction
	auto cond = ReqReset::Always;
	if(enumval(opts) & enumval(ReqOpts::Uniform)) cond |= ReqReset::Broadcast;
	LA.impl.rs_reset(req, cond, ReqStatus::Busy, ReqStatus::OK, request_tr);
	return std::move(*res);
}

// If actor-based request is used (`res_processor` is not given)
// then return delegated promise or request result inside `caf::result< req result type >`
// Otherwise, immediately start processing and pipe result into specified `res_processor`
template<typename R = void, typename F, typename... C>
auto request_data_impl(link_actor& LA, Req req, ReqOpts opts, F f_request, C... res_processor) {
	using rtraits = detail::request_traits<R, F>;
	using res_t = typename rtraits::res_t;
	using caf_res_t = caf::result<res_t>;

	static_assert(sizeof...(C) < 2, "Only one result processor is supported");

	const auto process_err = [&](auto&& er) -> res_t {
		if(er.ok()) {
			if constexpr(rtraits::can_invoke_inplace) {
				// invoke request inplace & update status
				auto res = rtraits::invoke_f_request(f_request, &LA);
				LA.impl.rs_update_from_data(res, enumval(opts & ReqOpts::Uniform));
				return res;
			}
			else
				return unexpected_err("Can't invoke request inplace");
		}
		else
			return res_t{tl::unexpect, er};
	};

	if(auto rworker = request_impl<R>(LA, req, opts, f_request)) {
		if constexpr(sizeof...(C) > 0) {
			using custom_rp_f = typename rtraits::custom_rp_f;
			(LA.send(*rworker, a_ack(), custom_rp_f{std::move(res_processor)}), ...);
		}
		else
			return caf_res_t{ LA.delegate(*rworker, a_ack()) };
	}
	else {
		auto res = process_err(rworker.error());
		if constexpr(sizeof...(C) > 0)
			(res_processor(std::move(res)), ...);
		else
			return caf_res_t{ std::move(res) };
	}
}

template<typename... C>
auto request_data(link_actor& LA, ReqOpts opts, C... res_processor) {
	return request_data_impl(
		LA, Req::Data, opts,
		[Limpl = LA.spimpl()] { return Limpl->data(); },
		std::move(res_processor)...
	);
}

template<typename... C>
auto request_data(unsafe_t, link_actor& LA, ReqOpts opts, C... res_processor) {
	return request_data_impl(
		LA, Req::Data, opts,
		[Limpl = LA.spimpl()]() -> obj_or_errbox { return Limpl->data(unsafe); },
		std::move(res_processor)...
	);
}

template<typename... C>
auto request_data_node(unsafe_t, link_actor& LA, ReqOpts opts, C... res_processor) {
	return request_data_impl(
		LA, Req::DataNode, opts,
		[Limpl = LA.spimpl()]() -> node_or_errbox { return Limpl->data_node(unsafe); },
		std::move(res_processor)...
	);
}

template<typename... C>
auto request_data_node(link_actor& LA, ReqOpts opts, C... res_processor) {
	return request_data_impl(
		LA, Req::DataNode, opts,
		[Limpl = LA.spimpl()]() -> node_or_errbox {
			return Limpl->data()
			.and_then([&](const auto& obj) -> node_or_err {
				if(obj) {
					if(auto n = obj->data_node())
						return n;
					return unexpected_err_quiet(Error::NotANode);
				}
				return unexpected_err_quiet(Error::EmptyData);
			});
		},
		std::move(res_processor)...
	);
}

NAMESPACE_END(blue_sky::tree)
