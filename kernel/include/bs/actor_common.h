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
#include <bs/timetypes.h>
#include <bs/error.h>
#include <bs/kernel/radio.h>
#include <bs/detail/tuple_utils.h>

#include <caf/fwd.hpp>
#include <caf/function_view.hpp>
#include <caf/event_based_actor.hpp>

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

/// @brief blocking invoke actor & return response like a function
/// @return always return `result_or_errbox<R>`
template<typename R, typename Actor, typename... Args>
auto actorf(caf::function_view<Actor>& factor, Args&&... args) {
	auto x = factor(std::forward<Args>(args)...);

	using T = typename decltype(x)::value_type;
	const auto extract_value = [&](auto& res) {
		// caf err passtrough
		if(!x) {
			res.emplace(tl::make_unexpected( forward_caf_error(x.error()) ));
			return;
		}

		if constexpr(std::is_same_v<T, caf::message>)
			x->extract({ [&](R& value) {
				res.emplace(std::move(value));
			} });
		else
			res.emplace(std::move(*x));
		if(!res) res.emplace( tl::make_unexpected(error{ "actorf: wrong result type R specified" }) );
	};

	if constexpr(tl::detail::is_expected<R>::value) {
		std::optional<R> res;
		extract_value(res);
		return std::move(*res);
	}
	else {
		std::optional<result_or_err<R>> res;
		extract_value(res);
		return std::move(*res);
	}
}

/// constructs function_view inside from passed handle & timeout
template<typename R, typename Actor, typename T, typename... Args>
auto actorf(const Actor& tgt, T timeout, Args&&... args) {
	return actorf<R>(
		caf::make_function_view(tgt, detail::cast_timeout(timeout)), std::forward<Args>(args)...
	);
}

/// operates on passed `scoped_actor` instead of function view
template<typename R, typename Actor, typename T, typename... Args>
auto actorf(const caf::scoped_actor& caller, const Actor& tgt, T timeout, Args&&... args) {
	auto res = [] {
		if constexpr(tl::detail::is_expected<R>::value)
			return std::optional<R>{};
		else
			return std::optional<result_or_err<R>>{};
	}();

	caller->request(tgt, detail::cast_timeout(timeout), std::forward<Args>(args)...)
	.receive(
		[&](R& value) { res.emplace(std::move(value)); },
		[&](const caf::error& er) { res.emplace(tl::make_unexpected(forward_caf_error(er))); }
	);
	return std::move(*res);
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

NAMESPACE_END(blue_sky)
