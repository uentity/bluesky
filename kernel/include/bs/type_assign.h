/// @author Alexander Gagarin (@uentity)
/// @date 22.01.2021
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include "error.h"
#include "propdict.h"
#include "kernel/errors.h"

NAMESPACE_BEGIN(blue_sky)
/*-----------------------------------------------------------------------------
 *  every BS type must be assignable - provide assing(target, source) definition
 *-----------------------------------------------------------------------------*/
using BS_TYPE_ASSIGN_FUN = error (*)(sp_obj /*target*/, sp_obj /*source*/, prop::propdict /*params*/);

/// Povide explicit specialization of this struct to disable assign (enabled by default for all types)
/// Also can be controlled by adding `T::bs_disable_assign` static bool constant
template<typename T, typename sfinae = void> struct has_disabled_assign : std::false_type {};

template<typename T> struct has_disabled_assign<T, std::void_t<decltype( T::bs_disable_assign )>> :
	std::bool_constant<T::bs_disable_assign> {};

NAMESPACE_BEGIN(detail)

template<typename T, typename = void>
struct assign_traits {
	// detect if type provide T::assign() member
	template<typename U, typename = void> struct has_member_assign : std::false_type {};
	template<typename U, typename = void> struct has_member_assign_wparams : std::false_type {};
	template<typename U> struct has_member_assign<U, std::void_t<decltype(
		std::declval<U&>().assign(std::declval<U&>())
	)>> : std::true_type {};
	template<typename U> struct has_member_assign_wparams<U, std::void_t<decltype(
		std::declval<U&>().assign(std::declval<U&>(), std::declval<prop::propdict>())
	)>> : std::true_type {};

	// if > 1 use `assign(source, params)`, otherwise `assign(source)`
	static constexpr char member = 2 * has_member_assign_wparams<T>::value + has_member_assign<T>::value;
	static constexpr auto noop = has_disabled_assign<T>::value && member == 0;
	static constexpr auto generic = !(member > 0 || noop);
};

inline auto noop_assigner(sp_obj, sp_obj, prop::propdict) -> error { return perfect; };

template<typename T>
auto make_assigner() {
	// [NOTE] strange if clause with extra constexpr if just to compile under VS
	if constexpr (assign_traits<T>::noop)
		return noop_assigner;
	else
		return [](sp_obj target, sp_obj source, prop::propdict params) -> error {
			// sanity
			if(!target) return error{"assign target", kernel::Error::BadObject};
			if(!source) return error{"assign source", kernel::Error::BadObject};
			auto& td = T::bs_type();
			if(!isinstance(target, td) || !isinstance(source, td))
				return error{"assign source or target", kernel::Error::UnexpectedObjectType};
			// invoke overload for type T
			return assign(static_cast<T&>(*target), static_cast<T&>(*source), std::move(params));
		};
}

NAMESPACE_END(detail)

/// Generic definition of assign() that works via operator=()
template<typename T>
auto assign(T& target, T& source, prop::propdict)
-> std::enable_if_t<detail::assign_traits<T>::generic, error> {
	static_assert(
		std::is_assignable_v<T&, T>,
		"Seems that type lacks default assignemnt operator. "
		"Either define it or provide overload of assign(T&, T&, prop::propdict) -> error"
	);
	if constexpr(std::is_nothrow_assignable_v<T, T>)
		target = source;
	else if constexpr(std::is_assignable_v<T, T>)
		return error::eval_safe([&] { target = source; });
	return perfect;
}

/// overload appears for types that have defined T::assign(const T&) method
template<typename T>
auto assign(T& target, T& source, prop::propdict params)
-> std::enable_if_t<detail::assign_traits<T>::member, error> {
	const auto invoke_assign = [&](auto&&... args) -> error {
		using R = std::remove_reference_t<decltype( target.assign(std::forward<decltype(args)>(args)...) )>;
		if constexpr(std::is_same_v<R, error>)
			return target.assign(std::forward<decltype(args)>(args)...);
		else
			return error::eval_safe([&] { target.assign(std::forward<decltype(args)>(args)...); });
	};

	if constexpr(detail::assign_traits<T>::member > 1)
		return invoke_assign(source, std::move(params));
	else
		return invoke_assign(source);
}

/// noop for types defined corresponding constant
template<typename T>
auto assign(T& target, T& source, prop::propdict)
-> std::enable_if_t<detail::assign_traits<T>::noop, error> {
	return perfect;
}

NAMESPACE_END(blue_sky)
