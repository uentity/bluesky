/// @file
/// @author uentity
/// @date 27.03.2019
/// @brief Function view (or function ref) to be used as arg type for passing callable
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
///
/// [NOTE] Credits to Simon Brand AKA TartanLlama & Vittorio Romeo
/// https://github.com/SuperV1234/vittorioromeo.info/blob/master/extra/passing_functions_to_functions/function_view.hpp
/// [NOTE] `function_view` DO NOT extend lifetime of passed callable
#pragma once

#include "../meta.h"

#include <caf/detail/type_list.hpp>
#include <functional>

namespace blue_sky {

/// generic declaration that later maatches callables
template<typename F> class function_view;

namespace detail {

/// traits to detect function_view instance
template<typename T> struct is_function_view : std::false_type {};
template<typename F> struct is_function_view< function_view<F> > : std::true_type {};

template<typename T> inline constexpr auto is_function_view_v =
	is_function_view< meta::remove_cvref_t<T> >::value;

/// treits for deducing callable signature
template<typename F, typename = void> struct deduce_callable {
	static_assert(std::is_invocable_v<F>, "Type isn't callable");
};
template<typename F> using deduce_callable_t = typename deduce_callable<std::remove_reference_t<F>>::type;

template<typename R, typename... Args>
struct deduce_callable<R (Args...), void> {
	using type = R (Args...);
	using args = caf::detail::type_list<Args...>;
	using result = R;
};
template<typename R, typename... Args>
struct deduce_callable<R (*)(Args...), void> {
	using type = R (Args...);
	using args = caf::detail::type_list<Args...>;
	using result = R;
};
template<typename C, typename R, typename... Args>
struct deduce_callable<R (C::*)(Args...), void> {
	using type = R (Args...);
	using args = caf::detail::type_list<Args...>;
	using result = R;
};
template<typename C, typename R, typename... Args>
struct deduce_callable<R (C::*)(Args...) const, void> {
	using type = R (Args...);
	using args = caf::detail::type_list<Args...>;
	using result = R;
};

template<typename F>
struct deduce_callable<F, std::void_t< decltype(&F::operator()) >> {
	using deducer = deduce_callable< std::remove_reference_t<decltype(&F::operator())> >;
	using type = typename deducer::type;
	using args = typename deducer::args;
	using result = typename deducer::result;
};

} // eof blue_sky::detail

template<typename F> using deduce_callable = detail::deduce_callable<std::remove_reference_t<F>>;
template<typename F> using deduce_callable_t = detail::deduce_callable_t<F>;

template<typename T> inline constexpr auto is_function_view_v = detail::is_function_view_v<T>;

/*-----------------------------------------------------------------------------
 *  function_view is a drop-in replacement for argument accepting any callable type
 *-----------------------------------------------------------------------------*/
template<typename R, typename... Args>
class function_view<R (Args...)> {
public:
	using result_t = R;
	using callable_t = R (Args...);
	using pointer_t = R (*)(Args...);
	using stdfn_t = std::function<callable_t>;

private:
	std::aligned_union_t<sizeof(void*), void*, pointer_t> fn_;
	R (*erased_fn_)(const void*, Args...);

	auto fn_mem() noexcept -> void* { return &fn_; }
	auto fn_mem() const noexcept -> const void* { return &fn_; }

	template<typename X> friend class function_view;

	template<typename F>
	inline constexpr auto init(F&& x) noexcept -> void {
		static_assert(
			std::is_invocable_r_v<R, F, Args...>,
			"Passed callable isn't compatible with signature of this function_view"
		);

		using Fpure = meta::remove_cvref_t<F>;
		if constexpr(
			std::is_constructible_v<pointer_t, F> || std::is_same_v<Fpure, stdfn_t>
		) {
			if constexpr(std::is_same_v<Fpure, stdfn_t>)
				// extract target function pointer from std::function
				::new (fn_mem()) pointer_t(*x.template target<pointer_t>());
			else
				// convert stateless callable (simple function or stateless lambda) to function pointer
				::new (fn_mem()) pointer_t(std::forward<F>(x));

			erased_fn_ = [](const void* fn_mem, Args... xs) -> R {
				pointer_t fn = *reinterpret_cast<const pointer_t*>(fn_mem);
				return std::invoke(fn, std::forward<Args>(xs)...);
			};
		}
		else {
			// generic case that can capture anything, but needs `x` to be alive
			::new (fn_mem()) (const void*)(std::addressof(x));
			erased_fn_ = [](const void* fn_mem, Args... xs) -> R {
				using ptr_t = void*;
				auto fn_ptr = *static_cast<const ptr_t*>(fn_mem);
				return std::invoke(
					*reinterpret_cast<std::add_pointer_t<F>>(fn_ptr), std::forward<Args>(xs)...
				);
			};
		}
	}

public:
	// construct from compatible callable
	template<typename F, typename = meta::enable_pf_ctor<function_view, F>>
	constexpr function_view(F&& x) noexcept { init(std::forward<F>(x)); }
	// move, copy ctors & assignment ops are default
	constexpr function_view(const function_view&) noexcept = default;
	constexpr function_view(function_view&&) noexcept = default;
	constexpr function_view& operator =(const function_view&) noexcept = default;
	constexpr function_view& operator =(function_view&&) noexcept = default;

	// assign from compatible callable
	template<typename F, typename = meta::enable_pf_ctor<function_view, F>>
	constexpr auto operator=(F&& x) noexcept -> function_view& {
		init(std::forward<F>(x));
		return *this;
	}

	constexpr friend auto swap(function_view& lhs, function_view& rhs) -> void {
		using std::swap;
		swap(lhs.fn_, rhs.fn_);
		swap(lhs.erased_fn_, rhs.erased_fn_);
	}

	// call stored callable with passed args
	template<typename... Ts>
	decltype(auto) operator()(Ts&&... xs) const
		noexcept(noexcept(erased_fn_(fn_mem(), std::forward<Ts>(xs)...)))
	{
		return erased_fn_(fn_mem(), std::forward<Ts>(xs)...);
	}
};

// deduction guides
template<typename F>
function_view(F) -> function_view< detail::deduce_callable_t<F> >;

} // eof blue_sky
