/// @author Alexander Gagarin (@uentity)
/// @date 03.02.2021
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../kernel/radio_subsyst.h"
#include <bs/detail/function_view.h>

#include <tuple>

NAMESPACE_BEGIN(blue_sky::python)

template<typename F, typename R, typename... Args, typename... LaunchAsync>
auto pipe_queue_impl(F f, const identity<R (Args...)> _, LaunchAsync... async_tag) {
	return [f = std::move(f)](Args&&... args) {
		KRADIO.enqueue(
			LaunchAsync{}..., [f, argtup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
				std::apply(f, std::move(argtup));
				return perfect;
			}
		);
	};
};


template<typename F, typename... LaunchAsync>
auto pipe_through_queue(F f, LaunchAsync... async_tag) {
	return pipe_queue_impl(std::move(f), identity< deduce_callable_t<F> >{}, async_tag...);
};


NAMESPACE_END(blue_sky::python)
