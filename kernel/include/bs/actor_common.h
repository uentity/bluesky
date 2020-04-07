/// @file
/// @author uentity
/// @date 02.12.2019
/// @brief Meta functions that provides better actor requests compatibility with BS types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/common.h>
#include <bs/atoms.h>
#include <bs/timetypes.h>
#include <bs/error.h>
#include <bs/kernel/radio.h>
#include <bs/detail/function_view.h>
#include <bs/detail/tuple_utils.h>

#include <caf/fwd.hpp>
#include <caf/function_view.hpp>
#include <caf/event_based_actor.hpp>
#include <caf/stateful_actor.hpp>
#include <caf/typed_behavior.hpp>

#include <optional>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(detail)

template<typename T>
constexpr auto cast_timeout(T t) {
	if constexpr(std::is_same_v<T, timespan>)
		return t != infinite ? caf::duration{t} : caf::infinite;
	else
		return t;
}

NAMESPACE_END(detail)

/// tag value for high priority messages
inline constexpr auto high_prio = caf::message_priority::high;

BS_API auto forward_caf_error(const caf::error& er) -> error;

/// obtain configured timeout for queries
BS_API auto def_timeout(bool for_long_task = false) -> caf::duration;

/// @brief blocking invoke actor & return response like a function
/// @return always return `result_or_errbox<R>`
template<typename R, typename Actor, typename... Args>
auto actorf(caf::function_view<Actor>& factor, Args&&... args) {
	// error is always transferred inside a box
	using R_ = std::conditional_t<std::is_same_v<R, error>, error::box, R>;
	// prepare result placeholder
	auto res = [] {
		if constexpr(tl::detail::is_expected<R_>::value)
			return std::optional<R_>{};
		else
			return std::optional<result_or_err<R_>>{};
	}();

	// make request & extract response
	if(auto x = factor(std::forward<Args>(args)...)) {
		using T = typename decltype(x)::value_type;
		if constexpr(std::is_same_v<T, caf::message>)
			x->extract({ [&](R_& value) {
				res.emplace(std::move(value));
			} });
		else
			res.emplace(std::move(*x));
		if(!res) res.emplace( tl::make_unexpected(error{ "actorf: wrong result type R specified" }) );
	}
	else // caf err passtrough
		res.emplace(tl::make_unexpected( forward_caf_error(x.error()) ));

	// if R is an error, then simply return `error` instead of `result_or_err<error>`
	if constexpr(std::is_same_v<R_, error::box>)
		return *res ? error{std::move(**res)} : std::move(*res).error();
	else
		return std::move(*res);
}

/// operates on passed `scoped_actor` instead of function view
template<typename R, typename Actor, typename T, typename... Args>
auto actorf(const caf::scoped_actor& caller, const Actor& tgt, T timeout, Args&&... args) {
	// error is always transferred inside a box
	using R_ = std::conditional_t<std::is_same_v<R, error>, error::box, R>;
	// prepare result placeholder
	auto res = [] {
		if constexpr(tl::detail::is_expected<R_>::value)
			return std::optional<R_>{};
		else
			return std::optional<result_or_err<R_>>{};
	}();

	// make request & extract response
	caller->request(tgt, detail::cast_timeout(timeout), std::forward<Args>(args)...)
	.receive(
		[&](R_& value) { res.emplace(std::move(value)); },
		[&](const caf::error& er) { res.emplace(tl::make_unexpected(forward_caf_error(er))); }
	);

	// if R is an error, then simply return `error` instead of `result_or_err<error>`
	if constexpr(std::is_same_v<R_, error::box>)
		return *res ? error{std::move(**res)} : std::move(*res).error();
	else
		return std::move(*res);
}

/// constructs scoped actor inside from passed handle & timeout
template<typename R, typename Actor, typename T, typename... Args>
auto actorf(const Actor& tgt, T timeout, Args&&... args) {
	return actorf<R>(
		caf::scoped_actor{kernel::radio::system()}, tgt, timeout, std::forward<Args>(args)...
	);
}

/// @brief spawn temp actor that makes specified request to `A` and pass result to callback `f`
template<caf::spawn_options Os = caf::no_spawn_options, typename Actor, typename F, typename... Args>
auto anon_request(Actor A, caf::duration timeout, bool high_priority, F f, Args&&... args) -> void {
	kernel::radio::system().spawn<Os>([
		high_priority, f = std::move(f), A = std::move(A), t = std::move(timeout),
		args = std::make_tuple(std::forward<Args>(args)...)
	] (caf::event_based_actor* self) mutable -> caf::behavior {
		std::apply([self, high_priority, A = std::move(A), t = std::move(t)](auto&&... args) {
			return high_priority ?
				self->request<caf::message_priority::high>(A, t, std::forward<decltype(args)>(args)...) :
				self->request<caf::message_priority::normal>(A, t, std::forward<decltype(args)>(args)...);
		}, std::move(args))
		.then(std::move(f));

		return {};
	});
}

/// @brief same as above but allows to receive result value returned from callback
template<caf::spawn_options Os = caf::no_spawn_options, typename Actor, typename F, typename... Args>
auto anon_request_result(Actor A, caf::duration timeout, bool high_priority, F f, Args&&... args) {
	struct rstate {
		// used to deliver result of `f()`
		caf::response_promise res;
	};
	// spawn worker actor that will make request and invoke `f`
	return kernel::radio::system().spawn<Os>([
		high_priority, f = std::move(f), A = std::move(A), t = std::move(timeout),
		args = std::make_tuple(std::forward<Args>(args)...)
	] (caf::stateful_actor<rstate>* self) mutable -> caf::behavior {
		// prepare response
		self->state.res = self->make_response_promise();
		// make request
		using Finfo = detail::deduce_callable<std::remove_reference_t<F>>;
		std::apply([self, high_priority, A = std::move(A), t = std::move(t)](auto&&... args) {
			return high_priority ?
				self->template request<high_prio>(A, t, std::forward<decltype(args)>(args)...) :
				self->template request(A, t, std::forward<decltype(args)>(args)...);
		}, std::move(args))
		.then(std::function<typename Finfo::type>{
			[res = self->state.res, f = std::move(f)](auto&&... xs) mutable {
				if constexpr(!std::is_same_v<typename Finfo::result, void>)
					res.deliver( f(std::forward<decltype(xs)>(xs)...) );
				else {
					f(std::forward<decltype(xs)>(xs)...);
					res.deliver(caf::unit);
				}
			}
		});

		return {
			// caller might send `a_ack` message to wait for callback invocation result
			[=](a_ack) { return self->state.res; }
		};
	});
}

/// @brief models 'or_else' for typed behaviors - make unified behavior from `first` and `second`
template<typename... SigsA, typename... SigsB>
auto first_then_second(caf::typed_behavior<SigsA...> first, caf::typed_behavior<SigsB...> second) {
	using namespace caf::detail;

	using SigsAB = type_list<SigsA..., SigsB...>;
	using result_t = tl_apply_t<tl_distinct_t<SigsAB>, caf::typed_behavior>;

	return result_t{
		typename result_t::unsafe_init(),
		caf::message_handler{ first.unbox().as_behavior_impl() }
		.or_else( second.unbox() )
	};
}

NAMESPACE_END(blue_sky)
