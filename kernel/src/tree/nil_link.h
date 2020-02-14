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
	static auto self() -> nil_link&;
	static auto stop() -> void;

	static auto pimpl() -> const sp_limpl&;
	static auto actor() -> const sp_ahandle&;

private:
	// remember raw pointer to impl for fast `is_nil()` checks
	sp_limpl pimpl_;
	sp_ahandle actor_;

	nil_link(sp_limpl pimpl);
	~nil_link();

	nil_link(const nil_link&) = delete;
	auto operator=(const nil_link&) -> nil_link& = delete;
};

NAMESPACE_END(blue_sky::tree)
