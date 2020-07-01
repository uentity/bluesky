/// @file
/// @author uentity
/// @date 01.07.2020
/// @brief engine::impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "engine_impl.h"

#include <bs/kernel/radio.h>

NAMESPACE_BEGIN(blue_sky::tree)
namespace kradio = kernel::radio;

auto engine::impl::factor(const void* L) -> sp_scoped_actor {
	// check if elem is already inserted and find insertion position
	{
		auto guard = guard_.lock(detail::shared);
		if(auto pf = rpool_.find(L); pf != rpool_.end())
			return pf->second;
	}
	// make insertion
	auto mguard = guard_.lock();
	return rpool_.try_emplace( L, std::make_shared<caf::scoped_actor>(kradio::system()) ).first->second;
}

auto engine::impl::release_factor(const void* L) -> void {
	auto guard = guard_.lock();
	rpool_.erase(L);
}

auto engine::impl::release_factors() -> void {
	auto guard = guard_.lock();
	rpool_.clear();
}


NAMESPACE_END(blue_sky::tree)
