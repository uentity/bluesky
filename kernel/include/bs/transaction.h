/// @date 01.09.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include "error.h"
#include "propdict.h"
#include "meta/variant.h"

#include <caf/allowed_unsafe_message_type.hpp>
#include <caf/event_based_actor.hpp>
#include <caf/result.hpp>

#include <cereal/cereal.hpp>

#include <optional>
#include <variant>

NAMESPACE_BEGIN(blue_sky)
/*-----------------------------------------------------------------------------
 *  define transaction result as variant of error & propdict with some useful methods
 *-----------------------------------------------------------------------------*/
// Why not just result_or_err<propdict>?
//
// Because error is 'normal' transaction result (it may not return additional info)
// Adding such errors using `tl::make_unexpected()` or via `unexpected_error()`
// is misleading.
// Constructor of `error` natively accepts `tr_result` and turns into succcess value if 'props' slot
// of result is filled.

struct tr_result : std::variant<prop::propdict, error> {
	using underlying_type = variant;
	using underlying_type::underlying_type;

	// construct from serializable `tr_result_box`
	using box = std::variant<prop::propdict, error::box>;

	tr_result(box tbox) {
		visit(meta::overloaded{
			[&](prop::propdict&& p) { emplace<0>(std::move(p)); },
			[&](error::box&& i) { emplace<1>(std::move(i)); }
		}, std::move(tbox));
	}

	// pack `tr_result` into `tr_result::box` suitable for serialization
	template<typename T>
	friend auto pack(T&& tres)
	-> std::enable_if_t<std::is_same_v<meta::remove_cvref_t<T>, tr_result>, box> {
		box res;
		visit(meta::overloaded{[&](auto&& v) {
			using V = decltype(v);
			using V_ = meta::remove_cvref_t<V>;
			if constexpr(std::is_same_v<V_, prop::propdict>)
				res.emplace<0>(std::forward<V>(v));
			else
				res.emplace<1>(std::forward<V>(v));
		}}, meta::forward_as<T, underlying_type>(tres));
		return res;
	}

	// check if return value carry props
	inline auto has_info() const -> bool { return index() == 0; }

	// extract props unchecked
	decltype(auto) info() const { return std::get<0>(*this); }
	decltype(auto) info() { return std::get<0>(*this); }
	// extract error unchecked
	decltype(auto) err() const { return std::get<1>(*this); }
	decltype(auto) err() { return std::get<1>(*this); }

	// check if transaction was successfull, i.e. error is OK or props passed
	operator bool() const { return has_info() ? true : err().ok(); }

	// checked info extractor: if rvalue passed in. then move error from it, otherwise copy
	template<typename T>
	friend auto extract_info(T&& tres)
	-> std::enable_if_t<std::is_same_v<meta::remove_cvref_t<T>, tr_result>, prop::propdict> {
		if(!tres.has_info()) return {};
		if constexpr(std::is_lvalue_reference_v<T>)
			return tres.info();
		else
			return std::move(tres.info());
	}

	// checked error extractor: if rvalue passed in. then move error from it, otherwise copy
	template<typename T>
	friend auto extract_err(T&& tres)
	-> std::enable_if_t<std::is_same_v<meta::remove_cvref_t<T>, tr_result>, error> {
		if(tres.has_info()) return perfect;
		if constexpr(std::is_lvalue_reference_v<T>)
			return tres.err();
		else
			return std::move(tres.err());
	}

	// implement map (like in `expected`)
	template<typename F>
	decltype(auto) map(F&& f) const {
		if(auto p = std::get_if<0>(this))
			return tr_result(f(*p));
		return *this;
	}

	template<typename F>
	decltype(auto) map(F&& f) {
		if(auto p = std::get_if<0>(this))
			return tr_result(f(*p));
		return *this;
	}

	template<typename F>
	decltype(auto) map_error(F&& f) const {
		if(auto e = std::get_if<1>(this))
			return tr_result(f(*e));
		return *this;
	}

	template<typename F>
	decltype(auto) map_error(F&& f) {
		if(auto e = std::get_if<1>(this))
			return tr_result(f(*e));
		return *this;
	}
};

/*-----------------------------------------------------------------------------
 *  transaction definition
 *-----------------------------------------------------------------------------*/
/// transaction is a function that is executed atomically in actor handler of corresponding object
template<typename R, typename... Ts> using transaction_t = std::function< R(Ts...) >;
/// async transaction takes actor pointer as 1st arg and can return result promise
template<typename R, typename... Ts>
using async_transaction_t = transaction_t<caf::result<typename R::box>, caf::event_based_actor*, Ts...>;
/// sum transaction type supported by object/link/node
template<typename R, typename T>
using sum_transaction_t = std::variant<
	transaction_t<R>, transaction_t<R, T>, async_transaction_t<R>, async_transaction_t<R, T>
>;

/// transaction accepted by misc object/link/node
using obj_transaction = sum_transaction_t<tr_result, sp_obj>;
using link_transaction = sum_transaction_t<error, tree::bare_link>;
using node_transaction = sum_transaction_t<error, tree::bare_node>;

/// transaction with no arguments
using transaction = transaction_t<tr_result>;
using async_transaction = async_transaction_t<tr_result>;
/// simple transaction returns error instead of extended transaction result
using simple_transaction = transaction_t<error>;
using async_simple_transaction = async_transaction_t<error>;

///////////////////////////////////////////////////////////////////////////////
//  traits
//
/// check if type is transaction
template<typename T>
struct is_transaction : public std::false_type {};

template<typename R, typename... Ts>
struct is_transaction<transaction_t<R, Ts...>> : public std::true_type {};

template<typename T> using is_transaction_t = is_transaction<meta::remove_cvref_t<T>>;
template<typename T> inline constexpr auto is_transaction_v = is_transaction_t<T>::value;

/// detect async transactions
template<typename T>
struct is_async_transaction : public std::false_type {};

template<typename R, typename... Ts>
struct is_async_transaction<transaction_t<R, Ts...>> : public std::conditional_t<
	std::is_same_v<meta::a1_t<Ts...>, caf::event_based_actor*>,
	std::true_type, std::false_type
> {};

template<typename T> using is_async_transaction_t = is_async_transaction<meta::remove_cvref_t<T>>;
template<typename T> inline constexpr auto is_async_transaction_v = is_async_transaction_t<T>::value;

/// detect sum transactions
template<typename T>
struct is_sum_transaction : public std::false_type {};

template<typename R, typename T>
struct is_sum_transaction<sum_transaction_t<R, T>> : public std::true_type {};

template<typename T> using is_sum_transaction_t = is_sum_transaction<meta::remove_cvref_t<T>>;
template<typename T> inline constexpr auto is_sum_transaction_v = is_sum_transaction_t<T>::value;

template<typename Tr, typename = std::enable_if_t< is_sum_transaction_v<Tr> >>
constexpr auto carry_async_transaction(const Tr& tr) -> bool {
	return tr.index() > 1;
}

///////////////////////////////////////////////////////////////////////////////
//  eval
//
/// run transaction & capture all exceptions into error slot of `tr_result`
template<typename R, typename... Ts, typename... Targs>
constexpr auto tr_eval(const transaction_t<R, Ts...>& tr, Targs&&... targs) {
	auto tres = std::optional<R>{};
	if(auto er = error::eval_safe([&] { tres.emplace( tr(std::forward<Targs>(targs)...) ); }))
		tres.emplace(std::move(er));
	return std::move(*tres);
}

/// evaluate sum transaction in actor context
template<typename Actor, typename R, typename T, typename F = noop_r_t<T>>
constexpr auto tr_eval(Actor* A, const sum_transaction_t<R, T>& tr, F&& target_getter = noop_r<T>()) {
	static_assert(std::is_invocable_r_v<T, F>);

	// result is expected to be `error` or `tr_result` => return packed
	using RB = typename R::box;
	using res_t = caf::result<RB>;

	return std::visit(meta::overloaded{
		[&](const transaction_t<R>& tr) -> res_t {
			return pack(tr_eval(tr));
		},

		[&](const async_transaction_t<R>& tr) -> res_t {
			return tr_eval(tr, A);
		},

		[&](const transaction_t<R, T>& tr) -> res_t {
			if(auto tgt = target_getter())
				return pack(tr_eval(tr, std::move(tgt)));
			return pack(error::quiet(Error::TrEmptyTarget));
		},

		[&](const async_transaction_t<R, T>& tr) -> res_t {
			if(auto tgt = target_getter())
				return tr_eval(tr, A, std::move(tgt));
			return pack(error::quiet(Error::TrEmptyTarget));
		}
	}, tr);
}

NAMESPACE_END(blue_sky)

/// mark all transaction types as non-serializable
NAMESPACE_BEGIN(caf)

template<typename R, typename... Ts>
struct allowed_unsafe_message_type<blue_sky::transaction_t<R, Ts...>> : std::true_type {};

template<typename R, typename T>
struct allowed_unsafe_message_type<blue_sky::sum_transaction_t<R, T>> : std::true_type {};

// make stringificator inspector happy
template<typename R, typename... Ts>
constexpr auto to_string(const blue_sky::transaction_t<R, Ts...>&) { return "transaction"; }

template<typename R, typename T>
constexpr auto to_string(const blue_sky::sum_transaction_t<R, T>&) { return "sum_transaction"; }

NAMESPACE_END(caf)

BS_ALLOW_VISIT(blue_sky::tr_result)
