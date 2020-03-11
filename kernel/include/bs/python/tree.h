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
PYBIND11_MAKE_OPAQUE(std::list<blue_sky::tree::sp_node>);

NAMESPACE_BEGIN(blue_sky::python)

/*-----------------------------------------------------------------------------
 *  trampoline for fusion_iface
 *-----------------------------------------------------------------------------*/
template<typename Fusion = tree::fusion_iface>
class py_fusion : public Fusion {
public:
	using Fusion::Fusion;

	auto populate(const tree::sp_node& root, const std::string& child_type_id = "") -> error override {
		PYBIND11_OVERLOAD_PURE(error, Fusion, populate, root, child_type_id);
	}

	auto pull_data(const sp_obj& root) -> error override {
		PYBIND11_OVERLOAD_PURE(error, Fusion, pull_data, root);
	}
};

NAMESPACE_END(blue_sky::python)
