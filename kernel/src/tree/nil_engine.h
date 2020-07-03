/// @file
/// @author uentity
/// @date 03.07.2020
/// @brief Link/node nil elements decl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/engine.h>

NAMESPACE_BEGIN(blue_sky::tree)

struct BS_HIDDEN_API nil_link {
	static auto nil_engine() -> const engine&;
	static auto pimpl() -> const engine::sp_engine_impl&;
	static auto actor() -> const engine::sp_ahandle&;

	static auto stop(bool wait_exit = false) -> void;

	struct self_actor;
	struct self_impl;

private:
	static auto reset() -> void;
};

struct BS_HIDDEN_API nil_node {
	static auto nil_engine() -> const engine&;
	static auto pimpl() -> const engine::sp_engine_impl&;
	static auto actor() -> const engine::sp_ahandle&;

	static auto stop(bool wait_exit = false) -> void;

	struct self_actor;
	struct self_impl;
	
private:
	static auto reset() -> void;
};

NAMESPACE_END(blue_sky::tree)
