/// @file
/// @author uentity
/// @date 15.02.2020
/// @brief Tweaked opaque-bindings for vector-like contsiners
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include <iterator>

NAMESPACE_BEGIN(blue_sky::python)
NAMESPACE_BEGIN(detail)

// detect if vector values needs to be copied when accessed from Vector
template<typename Vector>
struct vector_needs_copy : std::false_type {};

// ------- add proper accessors & some mmethods to opaque list-like class
template<typename Vector, typename PyVector>
auto rich_accessors(PyVector& cl) -> PyVector& {
	using T = typename Vector::value_type;
	using size_type = typename Vector::size_type;
	using SizeType = typename Vector::size_type;
	using DiffType = typename Vector::difference_type;
	using ItType   = typename Vector::iterator;

	static constexpr auto must_copy_values = vector_needs_copy<Vector>::value;
	static constexpr auto rvp =  must_copy_values ?
		py::return_value_policy::copy : py::return_value_policy::reference_internal;

	// deduce subscripting return value type
	using R = std::conditional_t<must_copy_values, T, T&>;

	static const auto wrap_i = [](DiffType i, SizeType n) {
		if (i < 0)
			i += n;
		if (i < 0 || (SizeType)i >= n)
			throw py::index_error();
		return i;
	};

	cl.def(py::init<size_type>());

	cl.def("erase", [](Vector &v, DiffType i) {
		i = wrap_i(i, v.size());
		v.erase(std::next(v.begin(), i));
	}, "erases element at index ``i``");

	cl.def("empty",     &Vector::empty, "checks whether the container is empty");
	cl.def("size",      &Vector::size,  "returns the number of elements");
	cl.def("push_back", (void (Vector::*)(const T&)) &Vector::push_back, "adds an element to the end");
	cl.def("pop_back",                             &Vector::pop_back, "removes the last element");

	cl.def("clear", &Vector::clear, "clears the contents");
	cl.def("swap",   &Vector::swap, "swaps the contents");

	cl.def("front", [](Vector &v) -> R {
		if (v.size()) return v.front();
		else throw py::index_error();
	}, rvp, "access the first element");

	cl.def("back", [](Vector &v) -> R {
		if (v.size()) return v.back();
		else throw py::index_error();
	}, rvp, "access the last element ");

	cl.def("__getitem__",
		[](Vector &v, DiffType i) -> R {
			i = wrap_i(i, v.size());
			return *std::next(v.begin(), i);
		},
		rvp
	);

	cl.def("__iter__",
		[](Vector &v) {
			return py::make_iterator<rvp, ItType, ItType, R>(
				v.begin(), v.end()
			);
		},
		py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
	);

	return cl;
}

NAMESPACE_END(detail)

template <typename Vector, typename holder_type = std::unique_ptr<Vector>, typename... Args>
auto bind_rich_vector(py::handle scope, std::string const &name, Args&&... args) {
	using Class_ = py::class_<Vector, holder_type>;

	// If the value_type is unregistered (e.g. a converting type) or is itself registered
	// module-local then make the vector binding module-local as well:
	using vtype = typename Vector::value_type;
	auto vtype_info = py::detail::get_type_info(typeid(vtype));
	bool local = !vtype_info || vtype_info->module_local;

	Class_ cl(scope, name.c_str(), pybind11::module_local(local), std::forward<Args>(args)...);

	// Declare the buffer interface if a buffer_protocol() is passed in
	py::detail::vector_buffer<Vector, Class_, Args...>(cl);

	cl.def(py::init<>());

	// Register copy constructor (if possible)
	py::detail::vector_if_copy_constructible<Vector, Class_>(cl);

	// Register comparison-related operators and functions (if possible)
	py::detail::vector_if_equal_operator<Vector, Class_>(cl);

	// Register stream insertion operator (if possible)
	py::detail::vector_if_insertion_operator<Vector, Class_>(cl, name);

	// Modifiers require copyable vector value type
	py::detail::vector_modifiers<Vector, Class_>(cl);

	// Accessor and iterator; return by value if copyable, otherwise we return by ref + keep-alive
	detail::rich_accessors<Vector, Class_>(cl);

	cl.def("__bool__",
		[](const Vector &v) -> bool {
			return !v.empty();
		},
		"Check whether the list is nonempty"
	);

	cl.def("__len__", &Vector::size);

	return cl;
}

NAMESPACE_END(blue_sky::python)
