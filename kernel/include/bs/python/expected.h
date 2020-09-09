/// @file
/// @author uentity
/// @date 17.09.2018
/// @brief Transparent conversion of tl::expected <-> Python
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <pybind11/pybind11.h>
#include <tl/expected.hpp>

NAMESPACE_BEGIN(pybind11::detail)
/*-----------------------------------------------------------------------------
 *  cast tl::expected <-> Python
 *-----------------------------------------------------------------------------*/
template<typename T, typename E>
struct type_caster< tl::expected<T, E> > {

	bool load(handle src, bool convert) {
		// first try to extract T
		auto caster = make_caster<T>();
		if(caster.load(src, convert)) {
			value = cast_op<T>(std::move(caster));
			return true;
		}
		// then E
		else {
			auto caster = make_caster<E>();
			if(caster.load(src, convert)) {
				value = tl::make_unexpected(cast_op<E>(std::move(caster)));
				return true;
			}
		}
		return false;
	}

	template<typename TT, typename EE, typename Expected>
	static handle cast_helper(Expected&& src, return_value_policy policy, handle parent) {
		if(src.has_value())
			return make_caster<TT>::cast(std::forward<TT>(src.value()), policy, parent);
		return make_caster<EE>::cast(std::forward<EE>(src.error()), policy, parent);
	}

	template<typename Expected>
	static handle cast(Expected&& src, return_value_policy policy, handle parent) {
		return cast_helper<T, E>(std::forward<Expected>(src), policy, parent);
	}

	using Type = tl::expected<T, E>;
	PYBIND11_TYPE_CASTER(Type, _("Expected[") +
		detail::concat(make_caster<T>::name, make_caster<E>::name) + _("]"));
};

// cast tl::monostate to None
template<> struct type_caster<tl::monostate> {
	using Type = tl::monostate;
	PYBIND11_TYPE_CASTER(Type, _("Monostate"));

	bool load(handle src, bool convert) {
		return src.is_none() ? true : false;
	}

	static handle cast(Type, return_value_policy, handle) {
		return pybind11::none();
	}
};

NAMESPACE_END(pybind11::detail)
