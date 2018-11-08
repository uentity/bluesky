/// @file
/// @author uentity
/// @date 07.11.2018
/// @brief Declare serialization support for Python types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <pybind11/pybind11.h>
#include "serialize.h"

BSS_FCN_DECL(serialize, pybind11::object)

NAMESPACE_BEGIN(cereal)

/*-----------------------------------------------------------------------------
 *  all types derived from `pybind11::object` are forwarded to base class
 *-----------------------------------------------------------------------------*/
template<typename Archive, typename T>
auto serialize(Archive& ar, T& t, std::uint32_t const /* version */) -> std::enable_if_t<
	std::is_base_of<pybind11::object, std::decay_t<T>>::value &&
	!std::is_same<pybind11::object, std::decay_t<T>>::value
> {
	ar(base_class<pybind11::object>(&t));
}

NAMESPACE_END(cereal)

