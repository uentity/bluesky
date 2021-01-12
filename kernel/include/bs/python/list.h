/// @file
/// @author uentity
/// @date 12.02.2019
/// @brief Allows to make opaque binding for std::set-like container
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "vector.h"

NAMESPACE_BEGIN(blue_sky::python)
NAMESPACE_BEGIN(detail)

template <typename List, typename Class_>
auto list_if_insertion_operator(Class_ &cl, std::string const &name)
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
void list_modifiers(std::enable_if_t<std::is_copy_constructible<typename List::value_type>::value, Class_> &cl) {
	using T = typename List::value_type;
	using SizeType = typename List::size_type;
	using DiffType = typename List::difference_type;

	static const auto wrap_i = [](DiffType i, SizeType n) {
		if (i < 0)
			i += n;
		if (i < 0 || (SizeType)i >= n)
			throw py::index_error();
		return i;
	};

	cl.def(py::init([](py::iterable it) {
		auto v = std::make_unique<List>();
		for (py::handle h : it)
			v->push_back(h.cast<T>());
		return v.release();
	}));

	cl.def("append",
		[](List &v, const T &value) { v.push_back(value); },
		py::arg("x"), "Add an item to the end of the list"
	);

	cl.def("extend",
		[](List &v, const List &src) {
			v.insert(v.end(), src.begin(), src.end());
		},
		py::arg("L"),
		"Extend the list by appending all the items in the given list"
	);

	cl.def("insert",
		[](List &v, DiffType i, const T &x) {
			// Can't use wrap_i; i == v.size() is OK
			if (i < 0)
				i += v.size();
			if (i < 0 || (SizeType)i > v.size())
				throw py::index_error();
			v.insert(std::next(v.begin(), i), x);
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
		[](List &v, DiffType i) {
			i = wrap_i(i, v.size());
			auto p_victim = std::next(v.begin(), i);
			T t = *p_victim;
			v.erase(p_victim);
			return t;
		},
		py::arg("i"),
		"Remove and return the item at index ``i``"
	);

	cl.def("__setitem__",
		[](List &v, DiffType i, const T &t) {
			i = wrap_i(i, v.size());
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
		[](List &v, DiffType i) {
			i = wrap_i(i, v.size());
			v.erase( std::next(v.begin(), i) );
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
	detail::list_if_insertion_operator<List, Class_>(cl, name);

	// Modifiers require copyable vector value type
	detail::list_modifiers<List, Class_>(cl);

	// Accessor and iterator; return by value if copyable, otherwise we return by ref + keep-alive
	detail::rich_accessors<List, Class_>(cl);

	cl.def("__bool__",
		[](const List &v) -> bool {
			return !v.empty();
		},
		"Check whether the list is nonempty"
	);

	cl.def("__len__", &List::size);

	return cl;
}

NAMESPACE_END(blue_sky::python)
