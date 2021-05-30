/// @file
/// @author uentity
/// @date 10.03.2019
/// @brief Define BS object 'property'
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "timetypes.h"
#include "uuid.h"
#include "meta.h"
#include "meta/is_shared_ptr.h"
#include "meta/variant.h"

#include <caf/detail/type_list.hpp>

#include <cstdint>
#include <iosfwd>
#include <map>
#include <variant>

NAMESPACE_BEGIN(blue_sky::prop)
// aliases for scalar types that property can handle
using integer   = std::int64_t;
using boolean   = bool;
using real      = double;
using timespan  = blue_sky::timespan;
using timestamp = blue_sky::timestamp;
using string    = std::string;
using object    = sp_obj;

class property;

NAMESPACE_BEGIN(detail)
/// enumerate scalars carried by BS property
using scalar_ts = caf::detail::type_list<
	// [NOTE] it's essential to place `bool` (and uuid) before `int`
	// because otherwise Python will cast all bools to int
	boolean, uuid, integer, real, timespan, timestamp, string, object
>;
inline constexpr auto scalar_ts_num = caf::detail::tl_size<scalar_ts>::value;

/// underlying type for list of scalars
template<typename T> struct scalar_list { using type = std::vector<T>; };
template<typename T> using scalar_list_t = typename scalar_list<T>::type;
using list_ts = caf::detail::tl_map_t<scalar_ts, scalar_list>;

/// underlying `std::variant` type of BS property
using carry_ts = caf::detail::tl_concat<scalar_ts, list_ts>::type;
using variant_prop_t = caf::detail::tl_apply_t<carry_ts, std::variant>;

/// helper compile-time checkers
template<typename T> inline constexpr auto is_prop_v = std::is_same_v<meta::remove_cvref_t<T>, property>;

template<typename T>
constexpr auto is_prop() {
	constexpr auto is_property = is_prop_v<T>;
	static_assert(is_property, "Argument isn't a blue-sky property instance");
	return is_property;
}

template<typename T> inline constexpr auto can_carry_scalar_v =
	caf::detail::tl_contains<detail::scalar_ts, T>::value;

template<typename T>
constexpr auto can_carry_scalar() {
	constexpr auto res = can_carry_scalar_v<T>;
	static_assert(res, "Property can't carry given scalar type");
	return res;
}

template<typename T> inline constexpr auto can_carry_list_v =
	caf::detail::tl_contains<detail::list_ts, T>::value;

template<typename T>
constexpr auto can_carry_list() {
	constexpr auto res = can_carry_list_v<T>;
	static_assert(res, "Property can't carry given list type");
	return res;
}

template<typename T> inline constexpr auto can_carry_type_v =
	caf::detail::tl_contains<detail::carry_ts, T>::value;

template<typename T>
constexpr auto can_carry_type() {
	constexpr auto res = can_carry_type_v<T>;
	static_assert(res, "Property can't carry given type");
	return res;
}

NAMESPACE_END(detail)

// tag for easier lists manipulation
template<typename T> using list_of = detail::scalar_list_t<T>;
template<typename T> using in_place_list_t = std::in_place_type_t<list_of<T>>;
template<typename T> inline constexpr in_place_list_t<T> in_place_list{};

/*-----------------------------------------------------------------------------
 *  BS property definition
 *-----------------------------------------------------------------------------*/
class property : public detail::variant_prop_t {
public:
	using underlying_type = detail::variant_prop_t;

private:
	// detect enums & all integer types
	template<typename T>
	static constexpr bool is_integer_ =
		std::is_enum_v<T> || (std::is_integral_v<T> && !std::is_same_v<T, bool>);

	template<typename T>
	static constexpr bool is_integer_v = is_integer_<meta::remove_cvref_t<T>>;

	// detect strings
	template<typename T>
	static constexpr bool is_string_v = !is_integer_v<T> &&
		std::is_convertible_v<meta::remove_cvref_t<T>, string>;

	// private ctors for cases when argument is {enum/integer, string, other}
	// public forwarding ctor performs dispatching to these
	// 1. encoding to allow dispatching: 0 - normal ctor, forward as is, 1 - enum/int, 2 - string
	template<int C> using ctor_arg_category = std::integral_constant<int, C>;
	// tag ctor argument with corresponding category
	template<typename T> using ctor_arg_tag = ctor_arg_category<is_integer_v<T> + 2*is_string_v<T>>;

	// 2. private ctors for each code
	// normal
	template<typename T>
	constexpr property(ctor_arg_category<0>, T&& value)
	noexcept(noexcept( underlying_type(std::declval<T>()) )) :
		underlying_type(std::forward<T>(value))
	{}
	// enum/int
	template<typename T>
	constexpr property(ctor_arg_category<1>, T value) noexcept :
		underlying_type(static_cast<integer>(value))
	{}
	// string
	template<typename T>
	constexpr property(ctor_arg_category<2>, T&& value)
	noexcept(noexcept( underlying_type(std::declval<std::in_place_type_t<string>>(), std::declval<T>()) )) :
		underlying_type(std::in_place_type_t<string>{}, std::forward<T>(value))
	{}

	template<typename T>
	using if_dispatch_allowed = std::enable_if_t<
		meta::enable_pf_ctor_v<property, T> &&
		(is_integer_v<T> || is_string_v<T> || std::is_constructible_v<underlying_type, T>)
	>;

public:
	using underlying_type::underlying_type;

	// explicit support for enums
	// + integral types & strings (required my MSVC)
	template<typename T, typename = if_dispatch_allowed<T>>
	constexpr property(T&& value)
	noexcept(noexcept( property(std::declval<ctor_arg_tag<T>>(), std::declval<T>()) )) :
		property(ctor_arg_tag<T>{}, std::forward<T>(value))
	{}

	property(const property&) = default;
	property(property&&) = default;
	auto operator =(const property&) -> property& = default;
	auto operator =(property&&) -> property& = default;

	template<typename T>
	constexpr auto operator=(T value) noexcept -> std::enable_if_t<is_integer_v<T>, property&> {
		emplace<integer>(static_cast<integer>(value));
		return *this;
	}

	template<typename T>
	constexpr auto operator=(T&& value) noexcept(noexcept( emplace<string>(std::declval<T>()) ))
	-> std::enable_if_t<is_string_v<T>, property&> {
		emplace<string>(std::forward<T>(value));
		return *this;
	}

	// improve init from initializer_list
	template<typename T>
	constexpr property(std::initializer_list<T> vlist) : underlying_type([&] {
		if constexpr(detail::can_carry_scalar_v<T>)
			return list_of<T>{ vlist };
		else if constexpr(std::is_convertible_v<T, string>)
			return list_of<std::string>(vlist.begin(), vlist.end());
		else if constexpr(std::is_integral_v<T>)
			return list_of<integer>(vlist.begin(), vlist.end());
		else if constexpr(std::is_arithmetic_v<T>)
			return list_of<real>(vlist.begin(), vlist.end());
		else if constexpr(meta::is_shared_ptr_v<T>) {
			using pointee = typename meta::is_shared_ptr<T>::pointee;
			if constexpr(std::is_convertible_v<pointee, objbase>)
				return list_of<sp_obj>(vlist.begin(), vlist.end());
			else {
				static_assert(
					meta::static_false<T>, "Property can't carry list of shared_ptrs to given type"
				);
				return underlying_type{};
			}
		}
		else {
			static_assert(meta::static_false<T>, "Property can't carry list of given values");
			return underlying_type{};
		}
	}()) {}
};

///////////////////////////////////////////////////////////////////////////////
//  helpers for manipulating property value
//
template<typename T, typename U>
constexpr decltype(auto) get(U&& p) {
	if constexpr(detail::is_prop<U>()) {
		if constexpr(std::is_enum_v<T>)
			return static_cast<T>(std::get<integer>(std::forward<U>(p)));
		else if constexpr(detail::can_carry_type<T>())
			return std::get<T>(std::forward<U>(p));
	}
}

template<std::size_t I, typename U>
constexpr decltype(auto) get(U&& p) {
	if constexpr(detail::is_prop<U>())
		return std::get<I>(std::forward<U>(p));
}

template<typename T, typename U>
constexpr auto get_if(U* p) noexcept {
	if constexpr(detail::is_prop<U>() && detail::can_carry_type<T>())
		return std::get_if<T>(p);
}

/// Intended to appear on right side, that's why const refs
template<typename T>
constexpr decltype(auto) get_or(const property* p, const T& def_value) noexcept {
	if constexpr(std::is_enum_v<T>) {
		auto pv = std::get_if<integer>(p);
		return pv ? static_cast<T>(*pv) : def_value;
	}
	else if constexpr(detail::can_carry_type<T>()) {
		auto pv = std::get_if<T>(p);
		return pv ? *pv : def_value;
	}
	else
		return def_value;
}

/// Tries to extract value of type `From`
/// If property doesn't hold such alternative,
/// return if value was found and updated
template<typename From = void, typename To = void>
constexpr bool extract(const property& source, To& target) noexcept {
	using carryT = std::conditional_t< std::is_same_v<From, void>, To, From >;
	if constexpr(detail::can_carry_type<carryT>()) {
		if(std::holds_alternative<carryT>(source)) {
			target = std::get<carryT>(source);
			return true;
		}
		return false;
	}
}

/// check if property holds 'None' value (null `sp_obj`)
constexpr bool is_none(const property& P) noexcept {
	return std::holds_alternative<object>(P) && !get<object>(P);
}

inline auto none() noexcept { return property(sp_obj()); }

/// custom analog of `std::visit` to fix compile issues with VS
template<
	typename F, typename P,
	typename = std::enable_if_t<std::is_same_v<meta::remove_cvref_t<P>, property>>
>
constexpr decltype(auto) visit(F&& f, P&& p) {
	// forward underlying type to std::visit
	using UT = typename std::remove_reference_t<P>::underlying_type;
	return std::visit(std::forward<F>(f), meta::forward_as<P, UT>(p));
}

///  formatting support
BS_API std::string to_string(const property& p);
BS_API std::ostream& operator <<(std::ostream& os, const property& x);

/// forward declarations for CAF type id block
class propdict;

/// propbook = map of props with given key type
template<typename Key> using propbook = std::map<Key, propdict, std::less<>>;
using propbook_s = propbook<std::string>;
using propbook_i = propbook<std::ptrdiff_t>;

NAMESPACE_END(blue_sky::prop)

BS_ALLOW_VISIT(blue_sky::prop::property)

/*-----------------------------------------------------------------------------
 *  CAF type id
 *-----------------------------------------------------------------------------*/
#include "atoms.h"

CAF_BEGIN_TYPE_ID_BLOCK(bs_props, blue_sky::detail::bs_props_cid_begin)

	CAF_ADD_TYPE_ID(bs_props, (blue_sky::prop::property))
	CAF_ADD_TYPE_ID(bs_props, (blue_sky::prop::propdict))
	CAF_ADD_TYPE_ID(bs_props, (blue_sky::prop::propbook_s))
	CAF_ADD_TYPE_ID(bs_props, (blue_sky::prop::propbook_i))

CAF_END_TYPE_ID_BLOCK(bs_props)
