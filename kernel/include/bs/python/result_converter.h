/// @file
/// @author uentity
/// @date 12.03.2020
/// @brief Make a pipe that converts result of given Python callable to specified type
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include <bs/detail/function_view.h>

NAMESPACE_BEGIN(blue_sky::python)

template<typename TR, typename F, typename R, typename... Args>
auto result_converter_impl(F&& f, TR if_none_v, identity<R (Args...)> _ = {}) {
	static_assert(std::is_same_v<R, py::object>, "Python callback must return py::object");
	return [f = std::forward<F>(f), if_none_v = std::move(if_none_v)](Args... args) mutable -> TR {
		auto guard = py::gil_scoped_acquire{};
		// detect if Python callback returns None
		auto res = f(std::forward<Args>(args)...);
		return res.is_none() ? std::move(if_none_v) : py::cast<TR>(std::move(res));
	};
};

/// produce functor that converts generic result of Python callback (py::object) to given C++ type
/// with passed None replacement value
template<typename TR, typename F>
auto make_result_converter(F&& f, TR if_none_v) {
	return result_converter_impl(
		std::forward<F>(f), std::move(if_none_v), identity< blue_sky::detail::deduce_callable_t<F> >{}
	);
};

NAMESPACE_END(blue_sky::python)
