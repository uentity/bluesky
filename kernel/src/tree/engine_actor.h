/// @author uentity
/// @date 13.11.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/engine.h>

#include <caf/actor_cast.hpp>
#include <caf/event_based_actor.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

struct BS_HIDDEN_API engine_actor_base : public caf::event_based_actor {
	using super = caf::event_based_actor;
	using sp_engine_impl = engine::sp_engine_impl;

	engine_actor_base(caf::actor_config& cfg, caf::group home, sp_engine_impl Eimpl);

	auto goodbye() -> void;

	auto on_exit() -> void override;

	// keeps strong pointer to impl
	sp_engine_impl pimpl_;
};

template<typename Item>
struct engine_actor : public engine_actor_base {
	using super = engine_actor_base;
	using type = typename Item::engine_actor;
	using engine_impl = typename Item::engine_impl;
	using actor_type = typename engine_impl::actor_type;
	using typed_behavior = typename actor_type::behavior_type;

	engine_actor(caf::actor_config& cfg, caf::group home, std::shared_ptr<engine_impl> Eimpl) :
		super(cfg, std::move(home), std::move(Eimpl)), impl([&]() -> auto& {
			return static_cast<engine_impl&>(*pimpl_);
		}())
	{}

	template<typename Actor>
	inline auto actor(Actor* A) -> actor_type {
		return caf::actor_cast<typename Actor::actor_type>(A);
	}

	// get typed node actor handle
	inline auto actor() -> actor_type {
		return actor(this);
	}

	template<typename EImpl = engine_impl>
	inline auto spimpl() const {
		return std::static_pointer_cast<EImpl>(pimpl_);
	}

	// impl ref for simpler access
	engine_impl& impl;
};

NAMESPACE_END(blue_sky::tree)
