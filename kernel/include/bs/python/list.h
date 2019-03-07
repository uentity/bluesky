/// @file
/// @author uentity
/// @date 12.02.2019
/// @brief Allows to make opaque binding for std::set-like container
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include <iterator>

NAMESPACE_BEGIN(blue_sky::python)
NAMESPACE_BEGIN(detail)

template <typename List, typename Class_> auto set_if_insertion_operator(Class_ &cl, std::string const &name)
	-> decltype(std::declval<std::ostream&>() << std::declval<typename List::value_type>(), void()) {
	using size_type = typename List::size_type;

	cl.def("__repr__",
		   [name](List &L) {
			std::ostringstream s;
			s << name << '[';
			bool first = true;
			for(const auto& x : L) {
				if(first) first = false;
				else s << ", ";
				s << x;
			}
			s << ']';
			return s.str();
		},
		"Return the canonical string representation of this list."
	);
}

// Vector modifiers -- requires a copyable vector_type:
// (Technically, some of these (pop and __delitem__) don't actually require copyability, but it seems
// silly to allow deletion but not insertion, so include them here too.)
template <typename List, typename Class_>
void set_modifiers(std::enable_if_t<std::is_copy_constructible<typename List::value_type>::value, Class_> &cl) {
	using T = typename List::value_type;
	using SizeType = typename List::size_type;
	using DiffType = typename List::difference_type;

	cl.def("append",
		   [](List &v, const T &value) { v.push_back(value); },
		   py::arg("x"),
		   "Add an item to the end of the list");

	cl.def(py::init([](py::iterable it) {
		auto v = std::make_unique<List>();
		for (py::handle h : it)
		   v->push_back(h.cast<T>());
		return v.release();
	}));

	cl.def("extend",
		[](List &v, const List &src) {
			v.insert(v.end(), src.begin(), src.end());
		},
		py::arg("L"),
		"Extend the list by appending all the items in the given list"
	);

	cl.def("insert",
		[](List &v, SizeType i, const T &x) {
			if (i > v.size())
				throw py::index_error();
			v.insert(std::next(v.begin(), (DiffType) i), x);
		},
		py::arg("i"), py::arg("x"),
		"Insert an item at a given position."
	);

	cl.def("pop",
		[](List &v) {
			if (v.empty())
				throw py::index_error();
			T t = v.back();
			v.pop_back();
			return t;
		},
		"Remove and return the last item"
	);

	cl.def("pop",
		[](List &v, SizeType i) {
			if (i >= v.size())
				throw py::index_error();
			auto p_victim = std::next(v.begin(), DiffType(i));
			T t = *p_victim;
			v.erase(p_victim);
			return t;
		},
		py::arg("i"),
		"Remove and return the item at index ``i``"
	);

	cl.def("__setitem__",
		[](List &v, SizeType i, const T &t) {
			if (i >= v.size())
				throw py::index_error();
			*std::next(v.begin(), i) = t;
		}
	);

	/// Slicing protocol
	cl.def("__getitem__",
		[](const List &v, py::slice slice) -> List * {
			size_t start, stop, step, slicelength;

			if (!slice.compute(v.size(), &start, &stop, &step, &slicelength))
				throw py::error_already_set();

			List *seq = new List();
			auto p_src = std::next(v.begin(), start);
			for (size_t i=0; i<slicelength; ++i) {
				seq->push_back(*p_src);
				std::advance(p_src, step);
			}
			return seq;
		},
		py::arg("s"),
		"Retrieve list elements using a slice object"
	);

	cl.def("__setitem__",
		[](List &v, py::slice slice,  const List &value) {
			size_t start, stop, step, slicelength;
			if (!slice.compute(v.size(), &start, &stop, &step, &slicelength))
				throw py::error_already_set();

			if (slicelength != value.size())
				throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");

			auto p_dest = std::next(v.begin(), start);
			auto p_src = value.begin();
			for (size_t i=0; i<slicelength; ++i) {
				*p_dest = *p_src;
				++p_src;
				std::advance(p_dest, step);
			}
		},
		"Assign list elements using a slice object"
	);

	cl.def("__delitem__",
		[](List &v, SizeType i) {
			if (i >= v.size())
				throw py::index_error();
			v.erase( std::next(v.begin(), DiffType(i)) );
		},
		"Delete the list elements at index ``i``"
	);

	cl.def("__delitem__",
		[](List &v, py::slice slice) {
			size_t start, stop, step, slicelength;

			if (!slice.compute(v.size(), &start, &stop, &step, &slicelength))
				throw py::error_already_set();

			for (size_t i = 0; i < slicelength; ++i) {
				v.erase( std::next(v.begin(), DiffType(start)) );
				start += step - 1;
			}
		},
		"Delete list elements using a slice object"
	);
}

// ------- add some mmethods to opaque list-like class
template<typename List, typename PyList>
auto make_rich_pylist(PyList& cl) -> PyList& {
	using size_type = typename List::size_type;
	using T = typename List::value_type;

	cl.def(py::init<size_type>());

	cl.def("erase",
		[](List &v, size_type i) {
		if (i >= v.size())
			throw py::index_error();
		v.erase(std::next(v.begin(), i));
	}, "erases element at index ``i``");

	cl.def("empty",         &List::empty,         "checks whether the container is empty");
	cl.def("size",          &List::size,          "returns the number of elements");
	cl.def("push_back", (void (List::*)(const T&)) &List::push_back, "adds an element to the end");
	cl.def("pop_back",                               &List::pop_back, "removes the last element");

	cl.def("clear", &List::clear, "clears the contents");
	cl.def("swap",   &List::swap, "swaps the contents");

	cl.def("front", [](List &v) {
		if (v.size()) return v.front();
		else throw py::index_error();
	}, "access the first element");

	cl.def("back", [](List &v) {
		if (v.size()) return v.back();
		else throw py::index_error();
	}, "access the last element ");

	return cl;
}

NAMESPACE_END(detail)

template <typename List, typename holder_type = std::unique_ptr<List>, typename... Args>
auto bind_list(py::handle scope, std::string const &name, Args&&... args) {
	using Class_ = py::class_<List, holder_type>;
	using T = typename List::value_type;
	using SizeType = typename List::size_type;
	using ItType   = typename List::iterator;

	// If the value_type is unregistered (e.g. a converting type) or is itself registered
	// module-local then make the vector binding module-local as well:
	using vtype = typename List::value_type;
	auto vtype_info = py::detail::get_type_info(typeid(vtype));
	bool local = !vtype_info || vtype_info->module_local;

	Class_ cl(scope, name.c_str(), pybind11::module_local(local), std::forward<Args>(args)...);

	cl.def(py::init<>());

	// Register copy constructor (if possible)
	py::detail::vector_if_copy_constructible<List, Class_>(cl);

	// Register comparison-related operators and functions (if possible)
	py::detail::vector_if_equal_operator<List, Class_>(cl);

	// Register stream insertion operator (if possible)
	detail::set_if_insertion_operator<List, Class_>(cl, name);

	// Modifiers require copyable vector value type
	detail::set_modifiers<List, Class_>(cl);

	// Accessor and iterator; return by value if copyable, otherwise we return by ref + keep-alive
	cl.def("__getitem__",
		[](List &v, SizeType i) -> T & {
			if (i >= v.size())
				throw py::index_error();
			return *std::next(v.begin(), i);
		},
		py::return_value_policy::reference_internal // ref + keepalive
	);

	cl.def("__iter__",
			[](List &v) {
				return py::make_iterator<
					py::return_value_policy::reference_internal, ItType, ItType, T&>(
					v.begin(), v.end());
			},
			py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
	);

	cl.def("__bool__",
		[](const List &v) -> bool {
			return !v.empty();
		},
		"Check whether the list is nonempty"
	);

	cl.def("__len__", &List::size);

	detail::make_rich_pylist<List>(cl);
	return cl;
}

NAMESPACE_END(blue_sky::python)
