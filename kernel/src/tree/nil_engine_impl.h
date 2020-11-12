/// @file
/// @author uentity
/// @date 03.07.2020
/// @brief Base class for nil handle impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/engine.h>
#include <bs/kernel/radio.h>

#include <caf/scoped_actor.hpp>
#include <caf/event_based_actor.hpp>

#include <atomic>

NAMESPACE_BEGIN(blue_sky::tree)

template<typename NilItem, typename ItemImpl>
struct nil_engine_impl : ItemImpl {
	using super = ItemImpl;
	using super::super;

	struct nil_engine : public engine {
		friend NilItem;
		using engine::engine;

		nil_engine(caf::actor engine_actor, sp_engine_impl pimpl) :
			engine(std::move(engine_actor), std::move(pimpl)),
			online_(bool(pimpl_))
		{}

		auto reset() -> void {
			online_ = false;
			static_cast<ItemImpl&>(*pimpl_).release_factors();
		}

		auto stop(bool wait_exit) -> void {
			if(online_) {
				auto nil_actor = raw_actor();
				auto waiter = caf::scoped_actor{kernel::radio::system(), false};
				waiter->send_exit(nil_actor, caf::exit_reason::kill);
				if(wait_exit)
					waiter->wait_for(nil_actor);
			}
		}

	private:
		std::atomic<bool> online_;
	};

	static auto internals() -> nil_engine& {
		static auto self_ = nil_engine(
			kernel::radio::system().spawn<typename NilItem::self_actor>(),
			std::make_shared<typename NilItem::self_impl>()
		);
		return self_;
	}
};


// base class for nil actors - prevent exit on misc events and allows only manual stopping
struct nil_engine_actor : public caf::event_based_actor {
	using super = caf::event_based_actor;
	nil_engine_actor(caf::actor_config& cfg);
};

NAMESPACE_END(blue_sky::tree)
