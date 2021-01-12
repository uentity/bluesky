/// @file
/// @author uentity
/// @date 15.01.2020
/// @brief Provide alternative impl for Python iterator that holds a container inside
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include <pybind11/detail/common.h>

NAMESPACE_BEGIN(blue_sky::python)
NAMESPACE_BEGIN(detail)

template <typename Container, typename Iterator>
struct iterator_state {
	Container c;
	Iterator it, end;
	bool first_or_done;

	iterator_state(Container&& c_) :
		c(std::move(c_)),
		it(std::begin(c)), end(std::end(c)),
		first_or_done(true)
	{}
	// state is move-only
	iterator_state(const iterator_state&) = delete;
	iterator_state(iterator_state&&) = default;
};

NAMESPACE_END(detail)

/// Captures container instance and makes an iterator for it
template<py::return_value_policy Policy = py::return_value_policy::reference_internal,
	typename Container, typename Iterator = decltype(std::begin(std::declval<Container>())),
	typename ValueType = decltype(*std::declval<Iterator>()),
	typename... Extra
>
py::iterator make_container_iterator(Container c, Extra &&... extra) {
	using state = detail::iterator_state<Container, Iterator>;

	if (!py::detail::get_type_info(typeid(state), false)) {
		py::class_<state>(py::handle(), "iterator", py::module_local())
			.def("__iter__", [](state &s) -> state& { return s; })
			.def("__next__", [](state &s) -> ValueType {
				if(!s.first_or_done)
					++s.it;
				else
					s.first_or_done = false;
				if(s.it == s.end) {
					s.first_or_done = true;
					throw py::stop_iteration();
				}
				return *s.it;
			}, std::forward<Extra>(extra)..., Policy);
	}

	return py::cast(state{std::move(c)}, py::return_value_policy::move);
}

NAMESPACE_END(blue_sky::python)
