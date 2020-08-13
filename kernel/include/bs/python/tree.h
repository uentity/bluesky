/// @file
/// @author uentity
/// @date 16.11.2017
/// @brief Trampolines and other public stuff for tree Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include <bs/tree/link.h>
#include <bs/tree/node.h>
#include <bs/tree/fusion.h>

// make it possible to bind opaque std::list & std::vector (w/o content copying)
PYBIND11_MAKE_OPAQUE(blue_sky::tree::links_v);
PYBIND11_MAKE_OPAQUE(std::list<blue_sky::tree::link>);
PYBIND11_MAKE_OPAQUE(std::list<blue_sky::tree::node>);

NAMESPACE_BEGIN(blue_sky::python)

// trampoline for fusion_iface
template<typename Fusion = tree::fusion_iface>
class py_fusion : public Fusion {
public:
	using Fusion::Fusion;

private:
	auto do_populate(sp_obj root, tree::link root_link, const std::string& child_type_id) -> error override {
		PYBIND11_OVERLOAD_PURE(error, Fusion, populate, std::move(root), root_link, child_type_id);
	}

	auto do_pull_data(sp_obj root, tree::link root_link) -> error override {
		PYBIND11_OVERLOAD_PURE(error, Fusion, pull_data, std::move(root), root_link);
	}
};

// gen bindings for engine::weak_ptr<T>
template<typename PyClass>
auto bind_weak_ptr(PyClass parent) {
	using T = typename PyClass::type;
	using weak_ptr_t = typename T::weak_ptr;
	return py::class_<weak_ptr_t>(parent, "weak_ptr")
		.def(py::init())
		.def(py::init<T>())

		.def(py::self == py::self)
		.def(py::self != py::self)
		.def(py::self < py::self)
		.def("__eq__",
			[](const weak_ptr_t& self, const T& other){ return self == other; },
			py::is_operator())
		.def("__neq__",
			[](const weak_ptr_t& self, const T& other){ return self != other; },
			py::is_operator()
		)

		.def("lock", &weak_ptr_t::lock, "Obtain strong reference")
		.def("expired", &weak_ptr_t::expired, "Check if this ptr is expired")
		.def("reset", &weak_ptr_t::reset, "Resets this ptr (makes it expired)")
	;
}

NAMESPACE_END(blue_sky::python)
