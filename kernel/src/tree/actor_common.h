/// @file
/// @author uentity
/// @date 14.07.2019
/// @brief Code shared among different BS actors
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/common.h>
#include <bs/timetypes.h>
#include <bs/error.h>
#include <bs/atoms.h>

#include <caf/function_view.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

inline constexpr auto def_data_timeout = timespan{ std::chrono::seconds(3) };

/// blocking invoke actor & return response like a function
/// [NOTE] always return `result_or_errbox<R>`
template<typename R, typename Actor, typename... Args>
inline auto actorf(caf::function_view<Actor>& factor, Args&&... args) {
	auto x = factor(std::forward<Args>(args)...);
	const auto x_err = [&] {
		return tl::make_unexpected(error{
			x.error().code(), factor.handle().home_system().render(x.error())
		});
	};

	using x_value_t = typename decltype(x)::value_type;
	if constexpr(tl::detail::is_expected<R>::value) {
		if(!x) return R{x_err()};
		if constexpr(std::is_same_v<x_value_t, caf::message>) {
			using T = typename R::value_type;
			R res;
			x->extract({ [&](T value) { res = std::move(value); } });
			return res;
		}
		else return std::move(*x);
	}
	else {
		using result_t = result_or_err<R>;
		if(!x) return result_t{x_err()};
		if constexpr(std::is_same_v<x_value_t, caf::message>) {
			result_t res;
			x->extract({ [&](R value) { res = std::move(value); } });
			return res;
		}
		else return result_t{ std::move(*x) };
	}
}

/// constructs function_view inside from passed handle & timeout
template<typename R, typename H, typename... Args>
inline auto actorf(const H& handle, blue_sky::timespan timeout, Args&&... args) {
	return actorf<R>(
		caf::make_function_view(handle, caf::duration{ timeout }), std::forward<Args>(args)...
	);
}

NAMESPACE_END(blue_sky::tree)
