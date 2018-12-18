/// @file
/// @author uentity
/// @date 16.08.2018
/// @brief Pattern of async API call using CAF actors
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../kernel.h"
#include <caf/scoped_actor.hpp>

namespace blue_sky { namespace detail {

template<class Derived>
class async_api_mixin {
public:
	using message_priority = caf::message_priority;
	explicit async_api_mixin() : sender_(BS_KERNEL.actor_system(), true) {}

	// link sender with target actor, so they die together
	auto init_sender() const -> void {
		sender_->link_to(derived().actor());
	}

	// pass any message to target actor
	template<message_priority P = message_priority::normal, typename... Args>
	auto send(Args&&... args) const {
		return sender_->send<P>(derived().actor(), std::forward<Args>(args)...);
	}

	auto sender() const -> const caf::scoped_actor& {
		return sender_;
	}

private:
	caf::scoped_actor sender_;
	inline auto derived() const -> const Derived& { return static_cast<const Derived&>(*this); }
};
	
}} /* namespace blue_sky::detail */

