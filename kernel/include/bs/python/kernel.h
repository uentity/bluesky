/// @file
/// @author uentity
/// @date 19.09.2017
/// @brief Custom bindings for kernel-related types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <pybind11/pybind11.h>
#include <bs/kernel.h>

NAMESPACE_BEGIN(pybind11) NAMESPACE_BEGIN(detail)
/*-----------------------------------------------------------------------------
 *  cast blue_sky::type_tuple <-> Python tuple
 *-----------------------------------------------------------------------------*/
template<> struct type_caster<blue_sky::type_tuple> {
	// carry resuling value
	using Type = blue_sky::type_tuple;
	bs_optional<Type> value;
	// if constructed from str name, we need to hold temp objects
	bs_optional<blue_sky::plugin_descriptor> pd_;
	bs_optional<blue_sky::type_descriptor> td_;

	// caster-related interface
	static constexpr auto name = _("BS type tuple");
	operator Type*() { return &value.value(); }
	operator Type&() { return value.value(); }
	operator Type() { return value.value(); }
	template <typename> using cast_op_type = Type;

	// helper to extract ref to given type from string or type instance
	// last parameter is holder for temp object initialized when string is passed
	template<typename T>
	static const T* load_str_or_T(handle src, bool convert, bs_optional<T>& tmp) {
		if(pybind11::isinstance<pybind11::str>(src)) {
			auto caster = make_caster<std::string>();
			if(caster.load(src, convert)) {
				tmp.emplace(cast_op<std::string>(caster).c_str());
				return &tmp.value();
			}
		}
		else {
			auto caster = make_caster<const T&>();
			if(caster.load(src, convert)) {
				return &cast_op<const T&>(caster);
			}
		}
		return nullptr;
	}

	bool load_from_value(handle src, bool convert) {
		// 1. try load type_descriptor
		if(auto p_td = load_str_or_T<blue_sky::type_descriptor>(src, convert, td_)) {
			value.emplace(*p_td);
			return true;
		}

		// 2. try load plugin_descriptor
		if(auto p_pd = load_str_or_T<blue_sky::plugin_descriptor>(src, convert, pd_)) {
			value.emplace(*p_pd);
			return true;
		}
		return false;
	}

	// convert from Python -> C++
	bool load(handle src, bool convert) {
		PyObject* pysrc = src.ptr();
		// 1. Check if tuple (plugin_descriptor, type_descriptor) is passed
		if(PyTuple_Check(pysrc)) {
			if(PyTuple_Size(pysrc) == 1) {
				return load_from_value(PyTuple_GetItem(pysrc, 0), convert);
			}
			if(PyTuple_Size(pysrc) > 1) {
				auto pd = load_str_or_T<blue_sky::plugin_descriptor>(PyTuple_GetItem(pysrc, 0), convert, pd_);
				auto td = load_str_or_T<blue_sky::type_descriptor>(PyTuple_GetItem(pysrc, 1), convert, td_);
				if(pd && td) {
					value.emplace(*pd, *td);
					return true;
				}
			}
			return false;
		}
		else {
			return load_from_value(src, convert);
		}
		return false;
	}

	// convert from C++ -> Python
	static handle cast(const Type& src, return_value_policy policy, handle parent) {
		return PyTuple_Pack(2,
			make_caster<typename Type::pd_t>::cast(src.pd(), policy, parent),
			make_caster<typename Type::td_t>::cast(src.td(), policy, parent)
		);
	}
};

NAMESPACE_END(detail) NAMESPACE_END(pybind11)

