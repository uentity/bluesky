/// @file
/// @author uentity
/// @date 13.02.2020
/// @brief Nil link declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "link_impl.h"

NAMESPACE_BEGIN(blue_sky::tree)

struct BS_HIDDEN_API nil_link {
	static auto pimpl() -> const sp_limpl&;
	static auto actor() -> const sp_ahandle&;
	static auto stop() -> void;
};

NAMESPACE_END(blue_sky::tree)
