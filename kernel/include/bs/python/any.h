/// @file
/// @author uentity
/// @date 26.09.2017
/// @brief Python binding for std::any
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <pybind11/pybind11.h>
#include <any>
#include <bs/fwd.h>
#define bs_any std::any

NAMESPACE_BEGIN(pybind11) NAMESPACE_BEGIN(detail)
/*-----------------------------------------------------------------------------
 *  cast blue_sky::type_tuple <-> Python tuple
 *-----------------------------------------------------------------------------*/
template<> struct type_caster<bs_any> {
	using Type = bs_any;
	using sp_obj = std::shared_ptr<blue_sky::objbase>;
	PYBIND11_TYPE_CASTER(Type, _("any"));

	bool load(handle src, bool convert) {
		namespace py = pybind11;
		if(py::isinstance<py::bool_>(src))
			value = py::cast<bool>(src);
		else if(py::isinstance<py::int_>(src))
			value = py::cast<long>(src);
		else if(py::isinstance<py::float_>(src))
			value = py::cast<double>(src);
		else if(py::isinstance<py::str>(src))
			value = py::cast<std::string>(src);
		else if(auto bsobj = py::cast<sp_obj>(src))
			value = std::move(src);
		else
			return false;
		return true;
	}

	template <typename PyType, typename U, typename... Us>
	static handle cast_alternative(bs_any& src, type_list<PyType, U, Us...>) {
		if(src.type() == typeid(U))
			return PyType(std::any_cast<U>(src)).release();
		return cast_alternative(src, type_list<PyType, Us...>());
	}
	template <typename PyType>
	static handle cast_alternative(bs_any& src, type_list<PyType>) {
		return none().release();
	}

	static handle cast(bs_any src, return_value_policy pol, handle parent) {
		namespace py = pybind11;
		const auto& src_t = src.type();

		// try load integer type
		handle simple = cast_alternative(src, type_list<
			py::int_, char, short, int, long, long long, unsigned int, unsigned long, unsigned long long
		>());
		if(!simple.is(none()))
			return simple;
		else if(src_t == typeid(bool))
			return bool_(std::any_cast<bool>(src));
		else if(!(simple = cast_alternative(src, type_list<py::float_, float, double>())).is(none()) ||
			!(simple = cast_alternative(src, type_list<py::str, std::string, const char*, char*>())).is(none())
		)
			return simple;
		else if(src_t == typeid(sp_obj)) {
			auto bs_caster = make_caster<sp_obj>();
			return bs_caster.cast(
				std::any_cast<sp_obj>(src),
				pol, parent
			);
		}
		throw py::value_error("bs_any can only store primitive types or BS objects");

		//// obtain most generic type caster
		//type_caster_generic tcg(src.type());
		//// obtain Python object from boost::any using registered Pybind11 type casters
		//const void* raw_src = boost::unsafe_any_cast<const void*>(&src);
		//if(!raw_src) return nullptr;
		//// expect only non-polymorphic types
		//std::pair<const void *, const type_info *> st = tcg.src_and_type(raw_src, src.type());
		//return tcg.cast(
		//	st.first, pol, parent, st.second,
		//	nullptr, nullptr
		//);
	}
};

NAMESPACE_END(detail) NAMESPACE_END(pybind11)

