/// @author uentity
/// @date 02.12.2019
/// @brief Meta functions that provides better actor requests compatibility with BS types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include "atoms.h"
#include "error.h"
#include "timetypes.h"
#include "transaction.h"
#include "kernel/radio.h"
#include "detail/function_view.h"
#include "detail/tuple_utils.h"

#include <caf/fwd.hpp>
#include <caf/function_view.hpp>
#include <caf/event_based_actor.hpp>
#include <caf/typed_actor.hpp>
#include <caf/typed_behavior.hpp>
#include <caf/is_actor_handle.hpp>
#include <caf/send.hpp>

#include <optional>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(detail)

template<typename T>
using if_actor_handle = std::enable_if_t<caf::is_actor_handle<T>::value>;

struct anon_sender {
	template<caf::message_priority P, typename... Ts>
	static auto send(Ts&&... xs) {
		caf::anon_send<P>(std::forward<Ts>(xs)...);
	};
};

template<typename T>
struct afres_keeper {
	// calc value type returned from request
	// `error` & `tr_result` are always transferred packed inside a box
	template<typename U> struct box_of : identity<typename U::box> {};
	static constexpr bool is_res_packed = std::is_same_v<T, error> || std::is_same_v<T, tr_result>;
	using R = typename std::conditional_t<is_res_packed, box_of<T>, identity<T>>::type;

	// setup placeholder for value returned from request (must have error attached)
	using P = std::conditional_t<
		is_res_packed || tl::detail::is_expected<T>::value || std::is_same_v<T, error::box>,
		std::optional<T>, std::optional<result_or_err<T>>
	>;
	P value;

	constexpr operator bool() const { return static_cast<bool>(value); }

	// forward -> value.emplace()
	template<typename U>
	constexpr decltype(auto) emplace(U&& x) {
		// if `value` carries expected & `x` is error, mark it as unexpected value
		if constexpr(
			std::is_same_v<meta::remove_cvref_t<U>, error> &&
			tl::detail::is_expected<typename P::value_type>::value
		)
			return value.emplace(tl::unexpect, std::forward<U>(x));
		else
			return value.emplace(std::forward<U>(x));
	}

	// move result out of placeholder
	decltype(auto) get() {
		// if value is empty, then paste error
		if(!value) emplace( error{ "actorf: wrong result type R specified" } );
		return std::move(*value);
	}
};

template<typename F>
struct closed_functor {
	template<typename FF>
	static auto make(FF f, caf::event_based_actor* self) {
		using namespace caf::detail;

		using Fargs = typename deduce_callable<F>::args;
		using x1_t = tl_head_t<Fargs>;
		constexpr bool add_self = std::is_pointer_v<x1_t> &&
			std::is_base_of_v<caf::event_based_actor, std::remove_pointer_t<x1_t>>;

		using res_args = std::conditional_t<add_self, tl_tail_t<Fargs>, Fargs>;
		return tl_apply_t<res_args, impl>::template make<add_self>(std::move(f), self);
	}

private:
	template<typename... Args>
	struct impl {
		template<bool AddSelf, typename FF>
		static auto make(FF f, caf::event_based_actor* self) {
			if constexpr(AddSelf)
				return [f = std::move(f), self](Args... xs) mutable {
					return f(self, std::forward<Args>(xs)...);
				};
			else if constexpr(std::is_same_v<F, FF>)
				return f;
			else
				return [f = std::move(f)](Args... xs) mutable {
					return f(std::forward<Args>(xs)...);
				};
		}
	};
};

template<typename... Ts> struct merge_with;

template<template<typename...> typename T, typename... SigsA, typename... SigsB>
struct merge_with<T<SigsA...>, T<SigsB...>> {
	using SigsAB = caf::detail::type_list<SigsA..., SigsB...>;
	using type = caf::detail::tl_apply_t<caf::detail::tl_distinct_t<SigsAB>, T>;
};

NAMESPACE_END(detail)

/// calculate merge result of typed actors or typed behaviors
/// in contrast to `extend_with<>` contains only unique entries
template<typename T, typename U> using merge_with = typename detail::merge_with<T, U>::type;

/// tag value for high priority messages
inline constexpr auto high_prio = caf::message_priority::high;

BS_API auto forward_caf_error(const caf::error& er, std::string_view msg = {}) -> error;

/// blocking invoke actor & return response like a function
template<typename R, typename Actor, typename... Args>
auto actorf(caf::function_view<Actor>& factor, Args&&... args) {
	// setup placeholder for request result
	using res_keeper = detail::afres_keeper<R>;
	using R_ = typename res_keeper::R;
	auto res = res_keeper{};

	// make request & extract response
	if(auto x = factor(std::forward<Args>(args)...)) {
		using T = typename decltype(x)::value_type;
		if constexpr(std::is_same_v<T, caf::message>) {
			auto extracter = caf::behavior{ [&](R_& value) {
				res.emplace(std::move(value));
			} };
			extracter(*x);
		}
		else
			res.emplace(std::move(*x));
	}
	else // caf err passtrough
		res.emplace(forward_caf_error(x.error()));

	return res.get();
}

/// operates on passed `scoped_actor` instead of function view
template<typename R, typename Actor, typename... Args>
auto actorf(const caf::scoped_actor& caller, const Actor& tgt, timespan timeout, Args&&... args) {
	// setup placeholder for request result
	using res_keeper = detail::afres_keeper<R>;
	using R_ = typename res_keeper::R;
	auto res = res_keeper{};

	// make request & extract response
	caller->request(tgt, timeout, std::forward<Args>(args)...)
	.receive(
		[&](R_& value) { res.emplace(std::move(value)); },
		[&](const caf::error& er) { res.emplace(forward_caf_error(er)); }
	);

	return res.get();
}

/// constructs scoped actor inside from passed handle & timeout
template<
	typename R, typename Actor, typename T, typename... Args,
	typename = detail::if_actor_handle<Actor>
>
auto actorf(const Actor& tgt, T timeout, Args&&... args) {
	return actorf<R>(
		caf::scoped_actor{kernel::radio::system()}, tgt, timeout, std::forward<Args>(args)...
	);
}

/// inplace match of given behavior and message, returns result of `sending` message to behavior
/// accepts void return type
template<typename R = void, typename... Args>
auto actorf(caf::behavior& bhv, Args&&... args) {
	// setup placeholder for request result
	auto res = [&] {
		if constexpr(std::is_same_v<R, void>)
			return std::optional<error>{};
		else
			return detail::afres_keeper<R>{};
	}();

	// setup value extracter from resulting message
	auto extracter = caf::message_handler{
		[&](const caf::error& er) { res.emplace(forward_caf_error(er)); }
	};
	if constexpr(!std::is_same_v<R, void>) {
		using R_ = typename decltype(res)::R;
		extracter = extracter.or_else(
			[&](R_& r) { res.emplace(std::move(r)); }
		);
	}

	// inplace match message against behavior
	auto m = caf::make_message(std::forward<Args>(args)...);
	if(auto req_res = bhv(m)) {
		auto extracter_bhv = caf::behavior{std::move(extracter)};
		extracter_bhv(*req_res);
	}
	else // if no answer returned => no match was found
		res.emplace(forward_caf_error(caf::sec::unexpected_message));

	if constexpr(!std::is_same_v<R, void>) return std::move(res.get());
}

/// @brief spawn temp actor that makes specified request to `A` and pass result to callback `f`
template<
	caf::spawn_options Os = caf::no_spawn_options,
	typename Actor, typename F, typename... Args,
	typename = detail::if_actor_handle<Actor>
>
auto anon_request(Actor A, timespan timeout, bool high_priority, F f, Args&&... args) -> void {
	kernel::radio::system().spawn<Os>([
		high_priority, A = std::move(A), t = timeout, f = std::move(f),
		args = std::make_tuple(std::forward<Args>(args)...)
	] (caf::event_based_actor* self) mutable {
		std::apply([self, high_priority, t, A = std::move(A)](auto&&... args) {
			return high_priority ?
				self->request<high_prio>(A, t, std::forward<decltype(args)>(args)...) :
				self->request(A, t, std::forward<decltype(args)>(args)...);
		}, std::move(args))
		.then(detail::closed_functor<F>::make(std::move(f), self));

		self->become({});
	});
}

/// @brief same as above but allows to receive result value returned from callback
template<
	caf::spawn_options Os = caf::no_spawn_options,
	typename Actor, typename F, typename... Args,
	typename = detail::if_actor_handle<Actor>
>
auto anon_request_result(Actor A, timespan timeout, bool high_priority, F f, Args&&... args) {
	// we need state to share lazily created response promise between `f` and `a_ack` query
	// seems that response promise must be created from message handler
	using Fres = typename deduce_callable<F>::result;
	struct rstate {
		using res_keeper_t = std::conditional_t<caf::is_result<Fres>::value, Fres, caf::result<Fres>>;
		std::optional<res_keeper_t> res;
	};

	// spawn worker actor that will make request and invoke `f`
	return kernel::radio::system().spawn<Os>([
		high_priority, A = std::move(A), t = timeout, f = std::move(f),
		args = std::make_tuple(std::forward<Args>(args)...)
	] (caf::stateful_actor<rstate>* self) mutable {
		// send `a_ack` request to obtain callback invocation result
		self->become(
			[=](a_ack) { return *self->state.res; }
		);
		// make request
		std::apply([self, high_priority, t, A = std::move(A)](auto&&... args) {
			return high_priority ?
				self->template request<high_prio>(A, t, std::forward<decltype(args)>(args)...) :
				self->request(A, t, std::forward<decltype(args)>(args)...);
		}, std::move(args))
		// [NOTE] use `await` to prevent `a_ack` processing while query processing not finished
		.await(detail::closed_functor<F>::make(
			[self, f = std::move(f)](auto&&... xs) mutable {
				if constexpr(!std::is_same_v<Fres, void>)
					self->state.res = f(std::forward<decltype(xs)>(xs)...);
				else {
					f(std::forward<decltype(xs)>(xs)...);
					self->state.res = caf::unit;
				}
			}, self
		));
	});
}

/// @brief models 'or_else' for typed behaviors - make unified behavior from `first` and `second`
template<typename... SigsA, typename... SigsB>
auto first_then_second(caf::typed_behavior<SigsA...> first, caf::typed_behavior<SigsB...> second) {
	using result_t = merge_with<decltype(first), decltype(second)>;
	return result_t{
		caf::unsafe_behavior_init,
		caf::message_handler{ first.unbox().as_behavior_impl() }
		.or_else( std::move(second.unbox()) )
	};
}

/// same as above, but adds typed behavior to type-erased one
template<typename... SigsB>
auto first_then_second(caf::behavior first, caf::typed_behavior<SigsB...> second) -> caf::behavior {
	return caf::message_handler{ first.unbox().as_behavior_impl() }
	.or_else( std::move(second.unbox()) );
}

/// @brief send to a group with compie-time check message against explicitly provided group typed iface
template<
	typename GroupActorType,
	caf::message_priority P = caf::message_priority::normal, typename ActorClass, typename... Ts
>
auto checked_send(ActorClass& src, const caf::group& dest, Ts&&... xs) -> void {
	// check if GroupSigs match with Ts
	static_assert(sizeof...(Ts) > 0, "no message to send");
	using args_token_t = caf::detail::type_list<caf::detail::strip_and_convert_t<Ts>...>;
	static_assert(
		caf::response_type_unbox<caf::signatures_of_t<GroupActorType>, args_token_t>::valid,
		"receiver does not accept given message"
	);
	// resort to actor's `send`
	src.template send<P>(dest, std::forward<Ts>(xs)...);
}

/// @brief same as above but with anon send (no source actor available)
template<
	typename GroupActorType,
	caf::message_priority P = caf::message_priority::normal, typename... Ts
>
auto checked_send(const caf::group& dest, Ts&&... xs) -> void {
	detail::anon_sender src;
	checked_send<GroupActorType, P>(src, dest, std::forward<Ts>(xs)...);
}

NAMESPACE_END(blue_sky)
