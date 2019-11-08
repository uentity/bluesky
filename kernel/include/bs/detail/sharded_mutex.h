/// @file
/// @author uentity
/// @date 07.11.2019
/// @brief Multiple mutexes + locks in one class
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../common.h"

#include <tuple>
#include <mutex>
#include <shared_mutex>

NAMESPACE_BEGIN(blue_sky::detail)

// disables locking at all
struct noop_mutex_tag {};
inline constexpr auto noop_mutex = noop_mutex_tag{};
// pass as first arg to produce shared lock
struct shared_tag {};
inline constexpr auto shared = shared_tag{};

template<typename M, typename... Ms>
struct sharded_mutex {
	template<size_t... Is>
	[[nodiscard]] constexpr auto lock() const {
		return lock_impl<LockMode::Unique, Is...>();
	}

	template<size_t... Is>
	[[nodiscard]] constexpr auto lock(shared_tag) const {
		return lock_impl<LockMode::Shared, Is...>();
	}

	using guards_t = std::tuple<M, Ms...>;
	mutable guards_t guards;

private:
	template<size_t I> using iconst_size_t = std::integral_constant<size_t, I>;

	// lock that does nothing
	struct noop_lock {
		template<typename... Ts> constexpr noop_lock(Ts&&... ts) noexcept {}
		// custom dtor to suppress unused variable warning
		~noop_lock() {};

		constexpr auto lock() const noexcept {}
		constexpr auto unlock() const noexcept {}
		template<typename... Ts> constexpr auto try_lock(Ts&&...) noexcept { return true; }
		template<typename... Ts> constexpr auto try_lock_for(Ts&&...) noexcept { return true; }
		template<typename... Ts> constexpr auto try_lock_until(Ts&&...) noexcept { return true; }

		constexpr operator bool() const noexcept { return true; }
		constexpr auto owns_lock() const noexcept { return true; }
	};

	enum class LockMode { Unique, Shared };

	// resolve lock type from mutex type
	template<typename Mi> using shared_lock_t = std::conditional_t<
		std::is_same_v<Mi, std::shared_mutex>, std::shared_lock<Mi>, noop_lock
	>;
	template<typename Mi> using unique_lock_t = std::conditional_t<
		std::is_same_v<Mi, std::shared_mutex>, std::unique_lock<Mi>, std::conditional_t<
			std::is_same_v<Mi, noop_mutex_tag>, noop_lock, std::lock_guard<Mi>
		>
	>;
	template<typename Mi, LockMode T> using lock_t = std::conditional_t< T == LockMode::Shared,
		shared_lock_t<std::remove_const_t<std::remove_reference_t<Mi>>>,
		unique_lock_t<std::remove_const_t<std::remove_reference_t<Mi>>>
	>;

	// remove indexes that refer to noop mutexes
	template<size_t... Rs, size_t I, size_t... Is>
	static constexpr auto filter_noops(std::index_sequence<Rs...> res, std::index_sequence<I, Is...> src) {
		if constexpr(std::is_base_of_v<noop_mutex_tag, std::tuple_element_t<I, guards_t>>)
			// skip Ith element
			return filter_noops(res, std::index_sequence<Is...>());
		else {
			// append Ith element
			return filter_noops(std::index_sequence<Rs..., I>(), std::index_sequence<Is...>());
			(void)res;
		}
	};
	// close recursion
	template<size_t... Rs>
	static constexpr auto filter_noops(std::index_sequence<Rs...> res, std::index_sequence<>) {
		return res;
	}

	// lock single mutex
	template<size_t I, LockMode T, bool Deffered = false>
	constexpr auto lock_impl() const {
		using lock_t = lock_t<std::tuple_element_t<I, guards_t>, T>;
		if constexpr(Deffered)
			return lock_t{ std::get<I>(guards), std::defer_lock };
		else
			return lock_t{ std::get<I>(guards) };
	}

	// lock one or several mutexes
	template<LockMode T, size_t I, size_t... In>
	constexpr auto lock_impl(std::index_sequence<I, In...> idx) const {
		if constexpr(sizeof...(In)) {
			// produce tuple of deferred locks and atomically lock 'em
			auto locks = std::make_tuple(lock_impl<I, T, true>(), lock_impl<In, T, true>()...);
			std::apply( [](auto&... Ls) { std::lock(Ls...); }, locks );
			return locks;
		}
		else
			return lock_impl<I, T>();
	}
	// for empty sequence return noop lock
	template<LockMode T>
	constexpr auto lock_impl(std::index_sequence<>) const {
		return noop_lock{};
	}

	// lock all mutexes
	template<LockMode T>
	constexpr auto lock_impl() const {
		return lock_impl<T>(filter_noops(
			std::index_sequence<>(), std::make_index_sequence< std::tuple_size_v<guards_t> >()
		));
	}

	// lock one or more mutexes
	template<LockMode T, size_t I1, size_t... In>
	constexpr auto lock_impl() const {
		return lock_impl<T>(filter_noops(
			std::index_sequence<>(), std::index_sequence<I1, In...>{}
		));
	}
};

template<typename M, typename = void>
struct sharded_same_mutex_t {
	using type = sharded_mutex<M>;
};

template<typename M, size_t... N>
struct sharded_same_mutex_t<M, std::index_sequence<N...>> {
	template<size_t I> using Me = M;
	using type = sharded_mutex<Me<N>...>;
};

// produce sharded mutex from N mutexes of same type
template<typename M, size_t N>
using sharded_same_mutex = typename sharded_same_mutex_t<M, std::make_index_sequence<N>>::type;

NAMESPACE_END(blue_sky::detail)
