/// @date 02.11.2020
/// @author uentity
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"

NAMESPACE_BEGIN(blue_sky::python)

template<typename PyEnum>
auto add_enumops(PyEnum& e) -> PyEnum& {
	using namespace pybind11;
	using E = typename PyEnum::type;
	using U = std::underlying_type_t<E>;
	constexpr bool is_convertible = std::is_convertible_v<E, U>;

	#define BSPY_ENUM_OP_CONV(op, expr)                  \
		e.attr(op) = cpp_function(                       \
			[](object a_, object b_) {                   \
				const auto a = static_cast<U>(int_(a_)); \
				const auto b = static_cast<U>(int_(b_)); \
				return static_cast<E>(expr);             \
			},                                           \
			name(op), is_method(e))

	// add missing bitwise ops that pybind11 doesn't add for enum classes
	if constexpr(!is_convertible) {
		BSPY_ENUM_OP_CONV("__and__",  a & b);
		BSPY_ENUM_OP_CONV("__rand__", a & b);
		BSPY_ENUM_OP_CONV("__or__",   a | b);
		BSPY_ENUM_OP_CONV("__ror__",  a | b);
		BSPY_ENUM_OP_CONV("__xor__",  a ^ b);
		BSPY_ENUM_OP_CONV("__rxor__", a ^ b);
		e.attr("__invert__") = cpp_function(
			[](py::object arg) { return static_cast<E>(~(static_cast<U>(int_(arg)))); },
			name("__invert__"), is_method(e)
		);
	}
	// additionally add plus/minus that can be useful with bitwise flags sometimes
	BSPY_ENUM_OP_CONV("__add__", a + b);
	BSPY_ENUM_OP_CONV("__radd__", a + b);
	BSPY_ENUM_OP_CONV("__sub__", a - b);
	BSPY_ENUM_OP_CONV("__rsub__", a - b);

	// allow initialization from int
	py::implicitly_convertible<int, E>();

	return e;
	#undef BSPY_ENUM_OP_CONV
}

template<typename E, typename... Extra>
auto bind_enum_with_ops(const py::handle& scope, const char* name, const Extra&... extra) {
	auto res = py::enum_<E>(scope, name, py::arithmetic(), extra...);
	return add_enumops(res);
};

NAMESPACE_END(blue_sky::python)
