/// @file
/// @author uentity
/// @date 26.06.2020
/// @brief Engine (actor + impl) part of tree link/node handle
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"

#include <caf/scoped_actor.hpp>

#include <functional>

NAMESPACE_BEGIN(blue_sky::tree)

class BS_API engine {
public:
	/// hidden handle that wraps strong ref to engine's internal actor
	/// + terminates it on destruction
	struct actor_handle;
	using sp_ahandle = std::shared_ptr<actor_handle>;

	/// tree element must inherit impl class from this one
	class impl;
	using sp_engine_impl = std::shared_ptr<impl>;

private:
	/// core of `weak_ptr`
	class weak_ptr_base {
	public:
		/// comparison
		auto operator==(const weak_ptr_base& rhs) const -> bool;
		auto operator!=(const weak_ptr_base& rhs) const -> bool;
		auto operator==(const engine& rhs) const -> bool;
		auto operator!=(const engine& rhs) const -> bool;

		/// ordering
		auto operator<(const weak_ptr_base& rhs) const -> bool;

		auto expired() const -> bool;

		/// resets ptr and makes it expired
		auto reset() -> void;

	protected:
		weak_ptr_base() = default;
		weak_ptr_base(const engine&);

		/// assign from engine-based tree handle
		auto assign(const engine& rhs) -> void;

		/// obtain engine from weak ptr
		auto lock() const -> engine;

	private:
		std::weak_ptr<actor_handle> actor_;
		std::weak_ptr<impl> pimpl_;
	};

public:
	/// holds weak ptr to tree item with semantics similar to `std::weak_ptr`
	template<typename Item>
	class weak_ptr : public weak_ptr_base {
	public:
		/// default constructor
		weak_ptr() = default;

		/// construct from item
		weak_ptr(const Item& src) : weak_ptr_base{src} {}

		/// assign from item
		auto operator=(const Item& rhs) -> weak_ptr& {
			weak_ptr_base::assign(rhs);
			return *this;
		}

		/// obtain item from weak ptr
		auto lock() const -> Item {
			return Item(weak_ptr_base::lock());
		}
	};

	/// special members
	engine(caf::actor engine_actor, sp_engine_impl pimpl);
	engine(sp_ahandle ah, sp_engine_impl pimpl);

	engine(const engine&) = default;
	auto operator=(const engine&) -> engine& = default;
	engine(engine&&) = default;
	auto operator=(engine&&) -> engine& = default;

	/// compare & sort support
	auto operator==(const engine& rhs) const -> bool;
	auto operator!=(const engine& rhs) const -> bool;
	auto operator<(const engine& rhs) const -> bool;
	auto operator==(const caf::actor& rhs) const -> bool;

	/// hash for appropriate containers
	auto hash() const noexcept -> std::size_t;

	/// check whether engine is valid
	auto has_engine() const noexcept -> bool;

	auto swap(engine& rhs) noexcept -> void;

	/*
	 * [NOTE] Despite that move semantics is supported by engine (for performance reasons),
	 * API below can only be invoked for VALID engine (`pimpl_` is non null), otherwise UB.
	 * Engine is intended to be used in derived tree handles that will always fallback to valid `nil` state.
	 */

	/// get engine's typed actor handle
	template<typename Handle>
	static auto actor(Handle&& H) {
		return caf::actor_cast<typename std::remove_reference_t<Handle>::actor_type>(H.raw_actor());
	}

	/// returns engine's string type ID
	auto type_id() const -> std::string_view;

	/// get engine's home group
	auto home() const -> const caf::group&;

	/// get engine's home group ID (empty for invalid / not started home)
	auto home_id() const -> std::string_view;

	/// unsubscribe event handler with known ID
	static auto unsubscribe(std::uint64_t event_cb_id) -> void;
	/// unsubscribe all self event handlers
	auto unsubscribe() const -> void;

protected:
	/// return engine's raw (dynamic-typed) actor handle
	// [NOTE] uncheked access
	auto raw_actor() const noexcept -> const caf::actor&;

	/// unconditional reset of actor_handle with new one
	auto install_raw_actor(caf::actor engine_actor) -> void;

	// strong ref to internal engine's actor
	// [NOTE] trick with shared ptr to handle is required to correctly track `engine` instances
	// and terminate internal actor when no more engines exist
	sp_ahandle actor_;

	// string ref to engine impl for fast unsafe access
	sp_engine_impl pimpl_;
};

NAMESPACE_END(blue_sky::tree)

NAMESPACE_BEGIN(std)

/// support for engines in hashed containers
template<> struct hash<::blue_sky::tree::engine> {
	auto operator()(const ::blue_sky::tree::engine& e) const noexcept { return e.hash(); }
};

NAMESPACE_END(std)
