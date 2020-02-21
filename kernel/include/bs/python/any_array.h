/// @file
/// @author uentity
/// @date 21.02.2020
/// @brief BS `any_array` opaque binder
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include "any.h"

NAMESPACE_BEGIN(blue_sky::python)

template< template<class> class array_traits >
void bind_any_array(py::module& m, const char* type_name) {
	using any_array_t = any_array<array_traits>;
	using key_type = typename any_array_t::key_type;
	using container_t = typename any_array_t::container_t;

	static auto get_size = [](const any_array_t& A){ return A.size(); };

	static auto make_array_key_iterator = [](any_array_t& A) {
		if constexpr(any_array_t::is_map)
			return py::make_key_iterator(A);
		else
			return py::make_iterator(A);
	};

	auto any_bind = py::class_<any_array_t>(m, type_name)
		.def(py::init<>())
		.def("__bool__",
			[](const any_array_t& A) -> bool { return !A.empty(); },
			"Check whether the array is nonempty"
		)
		.def("__len__", get_size)
		.def_property_readonly("size", get_size)

		.def("__contains__", [](const any_array_t& A, py::handle k) {
			auto key_caster = py::detail::make_caster<key_type>();
			return key_caster.load(k, true) ?
				A.has_key(py::detail::cast_op<key_type>(key_caster)) :
				false;
		})

		.def("__getitem__", [](any_array_t& A, const key_type& k) {
			using array_trait = typename any_array_t::trait;
			auto pval = array_trait::find(A, k);
			if(pval == A.end())
				throw py::key_error("There's no element with key = " + fmt::format("{}", k));
			return array_trait::iter2val(pval);
		}, py::return_value_policy::reference_internal)

		.def("__getitem__", [](any_array_t& A, std::size_t k) {
			if(k >= A.size())
				throw py::key_error("Index past array size");
			using array_trait = typename any_array_t::trait;
			return array_trait::iter2val(std::next(std::begin(A), k));
		}, py::return_value_policy::reference_internal)

		.def("__setitem__", [](any_array_t& A, const key_type& k, std::any value) {
			if(!any_array_t::trait::insert(A, k, std::move(value)).second)
				throw py::key_error("Cannot insert element with key = " + fmt::format("{}", k));
		})

		.def("__delitem__", [](any_array_t& m, const key_type& k) {
			auto it = any_array_t::trait::find(m, k);
			if (it == m.end()) throw py::key_error();
			m.erase(it);
		})

		.def("__iter__", make_array_key_iterator,
			py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
		)

		.def("items",
			[](any_array_t &m) { return py::make_iterator(m.begin(), m.end()); },
			py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
		)

		.def("keys", [](const any_array_t& A){ return A.keys(); })

		.def("values", [](const any_array_t& A) {
			std::vector<std::any> res;
			for(auto a = std::begin(A), end = std::end(A); a != end; ++a)
				res.push_back(any_array_t::trait::iter2val(a));
			return res;
		})
	;
}

NAMESPACE_END(blue_sky::python)
