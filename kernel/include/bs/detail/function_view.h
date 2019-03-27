/// @file
/// @author uentity
/// @date 27.03.2019
/// @brief Function view (or function ref) to be used as arg type for passing callable
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
//
// [NOTE] (C) Vittorio Romeo, https://github.com/SuperV1234/vittorioromeo.info/blob/master/extra/passing_functions_to_functions/function_view.hpp
#pragma once

#include <type_traits>
#include <functional>

namespace blue_sky {

/// [NOTE] `function_view` DO NOT extend lifetime of passed callable
template<typename F> class function_view;

template<typename R, typename... Args>
class function_view<R (Args...)> {
private:
	void* fn_;
	R (*erased_fn_)(void*, Args...);

public:
	using callable_t = R (Args...);
	using pointer_t = R (*)(Args...);
	using stdfn_t = std::function<callable_t>;

	// ctor accepts any compatible callable
	template<
		typename F, typename = std::enable_if_t<
			!std::is_same_v<std::decay_t<F>, function_view>
		>
	>
	function_view(F&& x) noexcept
		: fn_{(void*)std::addressof(x)}
	{
		constexpr auto is_compatible = std::is_invocable_r_v<R, F, Args...>;
		static_assert(is_compatible, "Passed callable isn't compatible with target function signature");
		if constexpr(is_compatible) {
			erased_fn_ = [](void* fn, Args... xs) -> R {
				return (*reinterpret_cast<std::add_pointer_t<F>>(fn))(
					std::forward<Args>(xs)...
				);
			};
		}
	}
	function_view(const function_view&) = default;
	function_view(function_view&&) = default;

	template<typename... Ts>
	decltype(auto) operator()(Ts&&... xs) const
		noexcept(noexcept(erased_fn_(fn_, std::forward<Ts>(xs)...)))
	{
		return erased_fn_(fn_, std::forward<Ts>(xs)...);
	}
};

} // eof blue_sky
