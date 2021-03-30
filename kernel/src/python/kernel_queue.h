/// @author Alexander Gagarin (@uentity)
/// @date 03.02.2021
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/python/common.h>
#include <bs/python/result_converter.h>
#include <bs/actor_common.h>
#include <bs/detail/function_view.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/propdict.h>

#include "../kernel/radio_subsyst.h"

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

// run any given callable through kernel's queue
template<typename F, typename... LaunchAsync>
auto pipe_through_queue(F f, LaunchAsync... async_tag) {
	return pipe_queue_impl(std::move(f), identity< deduce_callable_t<F> >{}, async_tag...);
};

// run Python transaction (applied to link/object) through queue
template<typename... Ts>
auto pytr_through_queue(std::function< py::object(Ts...) > tr) {
	return [tr = make_result_converter<tr_result>(std::move(tr), perfect)]
	(caf::event_based_actor* papa, Ts... args) mutable -> caf::result<tr_result::box> {
		// [NOTE] using request.await to stop messages processing while tr is executed
		auto res = papa->make_response_promise<tr_result::box>();
		papa->request(
			KRADIO.queue_actor(), kernel::radio::timeout(true),
			transaction{[tr = std::move(tr), argstup = std::make_tuple(std::forward<Ts>(args)...)]
			() mutable {
				return std::apply(std::move(tr), std::move(argstup));
			}}
		).await(
			[=](tr_result::box tres) mutable { res.deliver(std::move(tres)); },
			[=](const caf::error& er) mutable { res.deliver(pack(tr_result{ forward_caf_error(er) })); }
		);
		return res;
	};
}

NAMESPACE_END(blue_sky::python)
