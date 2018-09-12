/// @file
/// @author uentity
/// @date 15.08.2018
/// @brief Impl part of fusion_link PIMPL
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/kernel.h>
#include <bs/tree/fusion.h>
#include <bs/tree/node.h>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

struct fusion_link::impl {
	// bridge
	sp_fusion bridge_;
	// contained object
	sp_node data_;

	// ctor
	impl(sp_fusion&& bridge, sp_node&& data) :
		bridge_(std::move(bridge)), data_(std::move(data))
	{}
};

NAMESPACE_END(blue_sky) NAMESPACE_END(tree)

