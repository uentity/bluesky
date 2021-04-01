/// @file
/// @author uentity
/// @date 01.07.2020
/// @brief Describes engine::impl class
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/actor_common.h>
#include <bs/tree/engine.h>
#include <bs/tree/type_caf_id.h>
#include <bs/detail/sharded_mutex.h>

#include <caf/detail/shared_spinlock.hpp>
#include <caf/group.hpp>

#include <unordered_map>

// helper macro to inject engine type ids
#define ENGINE_TYPE_DECL                           \
static auto type_id_() -> std::string_view;        \
auto type_id() const -> std::string_view override;

#define ENGINE_TYPE_DEF(eimpl_class, typename)                               \
auto eimpl_class::type_id_() -> std::string_view { return typename; }        \
auto eimpl_class::type_id() const -> std::string_view { return type_id_(); }

NAMESPACE_BEGIN(blue_sky::tree)

using engine_impl_mutex = caf::detail::shared_spinlock;

/// tree element must inherit impl class from this one
class engine::impl {
public:
	// mixin to obtain Item instance from impl
	template<typename Item>
	class access {
	public:
		auto super_engine() const { return super_.lock(); }

	protected:
		friend Item;

		auto reset_super_engine(const Item& x) { super_ = x; }

	protected:
		engine::weak_ptr<Item> super_;
	};

	/// get downcasted impl realization from handle
	template<typename EngineImpl, typename Handle>
	static decltype(auto) pimpl(const Handle& H) {
		if constexpr(std::is_base_of_v<engine, Handle>)
			return static_cast<EngineImpl&>(*static_cast<const engine&>(H).pimpl_);
		else if constexpr(std::is_base_of_v<engine::impl, Handle>)
			return static_cast<const EngineImpl&>(H);
	}

	template<typename Handle>
	static decltype(auto) pimpl(const Handle& H) {
		if constexpr(std::is_base_of_v<engine, Handle>)
			return pimpl<typename Handle::engine_impl>(H);
		else if constexpr(std::is_base_of_v<engine::impl, Handle>)
			return H;
	}

private:
	template<typename Handle>
	using if_engine_handle = std::enable_if_t<std::is_base_of_v<engine, Handle>>;

	template<typename Handle>
	using engine_impl_t = std::remove_reference_t<decltype( pimpl(std::declval<Handle>()) )>;

public:
	using sp_engine_impl = std::shared_ptr<impl>;
	using sp_scoped_actor = std::shared_ptr<caf::scoped_actor>;

	/// engine home group
	caf::group home;

	/// get engine's home group ID (empty for invalid / not started home)
	auto home_id() const -> std::string_view;

	auto swap(impl& rhs) -> void;

	// required, `engine` holds a pointer to `impl`
	virtual ~impl() = default;

	/// return engine's type ID
	virtual auto type_id() const -> std::string_view = 0;

	/// can be shadowed in derived handle impl class to further customize timeouts
	inline auto timeout(bool for_long_task) const {
		return kernel::radio::timeout(for_long_task);
	}

	/// actorf() will resolve actor type using this function
	template<typename Handle>
	static auto actor(const Handle& L) {
		// [NOTE] return private (extended) interface
		return caf::actor_cast<typename Handle::engine_impl::actor_type>(L.raw_actor());
	}

	/// make request to engine via tree item handle H
	template<typename R, typename Handle, typename... Args, typename = if_engine_handle<Handle>>
	static auto actorf(const Handle& H, timespan timeout, Args&&... args) {
		return blue_sky::actorf<R>(
			Handle::engine_impl::actor(H), timeout, std::forward<Args>(args)...
		);
	}

	/// same as above, but substitute configured timeouts
	template<typename R, typename Handle, typename... Args, typename = if_engine_handle<Handle>>
	static auto actorf(const Handle& H, Args&&... args) {
		return actorf<R>(H, pimpl(H).timeout(false), std::forward<Args>(args)...);
	}

	template<typename R, typename Handle, typename... Args, typename = if_engine_handle<Handle>>
	static auto actorf(long_op_t, const Handle& H, Args&&... args) {
		return actorf<R>(H, pimpl(H).timeout(true), std::forward<Args>(args)...);
	}

	/// send message to home group with compile-time check against home actor type
	template<
		caf::message_priority P = caf::message_priority::normal, typename ActorClass, typename... Ts
	>
	static auto send_home(ActorClass* src, Ts&&... xs) -> void {
		using home_actor_t = typename engine_impl_t<decltype(src->impl)>::home_actor_type;
		checked_send<home_actor_t, P>(*src, src->impl.home, std::forward<Ts>(xs)...);
	}

	/// same as above but accepts Handle (or it's impl) instead of engine actor
	template<
		caf::message_priority P = caf::message_priority::normal, typename Handle, typename... Ts
	>
	static auto send_home(const Handle& H, Ts&&... xs) -> void {
		using home_actor_t = typename engine_impl_t<Handle>::home_actor_type;
		checked_send<home_actor_t, P>(pimpl(H).home, std::forward<Ts>(xs)...);
	}
};

NAMESPACE_END(blue_sky::tree)
