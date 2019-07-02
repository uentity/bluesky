/// @file
/// @author uentity
/// @date 27.03.2019
/// @brief Function view (or function ref) to be used as arg type for passing callable
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
//
// [NOTE] Heavily modified version by Vittorio Romeo,
// https://github.com/SuperV1234/vittorioromeo.info/blob/master/extra/passing_functions_to_functions/function_view.hpp
#pragma once

#include <type_traits>
#include <functional>

namespace blue_sky {

/// [NOTE] `function_view` DO NOT extend lifetime of passed callable
template<typename F> class function_view;

template<typename R, typename... Args>
class function_view<R (Args...)> {
public:
	using callable_t = R (Args...);
	using pointer_t = R (*)(Args...);
	using stdfn_t = std::function<callable_t>;

private:
	void* fn_;
	R (*erased_fn_)(void*, Args...);

	template<typename F>
	inline constexpr auto init(F&& x) noexcept -> void {
		constexpr auto is_compatible = std::is_invocable_r_v<R, F, Args...>;
		static_assert(is_compatible, "Passed callable isn't compatible with target function signature");

		if constexpr(is_compatible) {
			if constexpr(std::is_same_v<std::decay_t<F>, stdfn_t>) {
				fn_ = (void*)(*std::forward<F>(x).template target<pointer_t>());
				erased_fn_ = [](void* fn, Args... xs) -> R {
					return std::invoke(
						reinterpret_cast<pointer_t>(fn), std::forward<Args>(xs)...
					);
				};
			}
			else {
				fn_ = (void*)std::addressof(x);
				erased_fn_ = [](void* fn, Args... xs) -> R {
					return std::invoke(
						*reinterpret_cast<std::add_pointer_t<F>>(fn), std::forward<Args>(xs)...
					);
				};
			}
		}
	}

public:
	// construct from compatible callable
	template<typename F, typename = std::enable_if_t<
		!std::is_same_v<std::decay_t<F>, function_view>
	>>
	constexpr function_view(F&& x) noexcept { init(std::forward<F>(x)); }
	// move, copy ctors & assignment ops are default
	constexpr function_view(const function_view&) noexcept = default;
	constexpr function_view(function_view&&) noexcept = default;
	constexpr function_view& operator =(const function_view&) noexcept = default;
	constexpr function_view& operator =(function_view&&) noexcept = default;

	// assign from compatible callable
	template<typename F, typename = std::enable_if_t<
		!std::is_same_v<std::decay_t<F>, function_view>
	>>
	constexpr auto operator=(F&& x) noexcept -> function_view& {
		init(std::forward<F>(x));
		return *this;
	}

	constexpr auto swap(function_view& rhs) -> void {
		std::swap(fn_, rhs.fn_);
		std::swap(erased_fn_, rhs.erased_fn_);
	}

	// call stored callable with passed args
	template<typename... Ts>
	decltype(auto) operator()(Ts&&... xs) const
		noexcept(noexcept(erased_fn_(fn_, std::forward<Ts>(xs)...)))
	{
		return erased_fn_(fn_, std::forward<Ts>(xs)...);
	}
};

// swap support
template<typename R, typename... Args>
constexpr auto swap(function_view<R (Args...)>& lhs, function_view<R (Args...)>& rhs) -> void {
	lhs.swap(rhs);
}

// deduction guides
namespace detail {

template<typename F, typename = void> struct deduce_callable {
	static_assert(std::is_invocable_v<F>, "Type isn't callable");
};
template<typename F> using deduce_callable_t = typename deduce_callable<std::remove_reference_t<F>>::type;

template<typename R, typename... Args>
struct deduce_callable<R (Args...), void> {
	using type = R (Args...);
};
template<typename R, typename... Args>
struct deduce_callable<R (*)(Args...), void> {
	using type = R (Args...);
};
template<typename C, typename R, typename... Args>
struct deduce_callable<R (C::*)(Args...), void> {
	using type = R (Args...);
};
template<typename C, typename R, typename... Args>
struct deduce_callable<R (C::*)(Args...) const, void> {
	using type = R (Args...);
};

template<typename F>
struct deduce_callable<F, std::void_t< decltype(&F::operator()) >> {
	using type = deduce_callable_t< decltype(&F::operator()) >;
};

} // eof blue_sky::detail

template<typename F>
function_view(F) -> function_view< detail::deduce_callable_t<F> >;

} // eof blue_sky
