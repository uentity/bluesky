/// @file
/// @author uentity
/// @date 18.08.2016
/// @brief Contains BlueSky error reporting subsystem
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include "meta.h"
#include "meta/variant.h"
#include <tl/expected.hpp>

#include <optional>
#include <system_error>
#include <iosfwd>

#define BS_REGISTER_ERROR_ENUM(E) \
namespace std { template<> struct is_error_code_enum< E > : true_type {}; }

#define EVAL if(auto er = ::blue_sky::error::eval(
#define EVAL_SAFE if(auto er = ::blue_sky::error::eval_safe(
#define EVAL_SAFE_QUIET if(auto er = ::blue_sky::error::eval_safe_quiet(
#define RETURN_EVAL_ERR )) return er;

#define SCOPE_EVAL if(auto er = ::blue_sky::error::eval([&] {
#define SCOPE_EVAL_SAFE if(auto er = ::blue_sky::error::eval_safe([&] {
#define SCOPE_EVAL_SAFE_QUIET if(auto er = ::blue_sky::error::eval_safe_quiet([&] {
#define RETURN_SCOPE_ERR })) return er;

/*-----------------------------------------------------------------------------
 *  define default error codes: OK and not OK
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN(blue_sky)

/// error code that is assigned by default if not specified explicitly
enum class Error {
	OK = 0,
	Happened = -1,
	Undefined = -2 // will transform to OK or Happened depending on quiet status
};
BS_API std::error_code make_error_code(Error);

NAMESPACE_END(blue_sky)

// register default error code
BS_REGISTER_ERROR_ENUM(blue_sky::Error)

NAMESPACE_BEGIN(blue_sky)
/*-----------------------------------------------------------------------------
 *  error class decalration
 *-----------------------------------------------------------------------------*/
class BS_API error {
public:
	/// allows to make quiet error from boolean state: true means success, false - fail
	struct quiet_flag {
		const bool value;

		// can only be initialized from exactly bool value
		template<typename T, typename = std::enable_if_t<std::is_same_v<T, bool>>>
		constexpr quiet_flag(T v) : value(v) {}
	};

	/// allows to make error from int code
	struct raw_code {
		const int value;

		// can only be initialized from integral non-bool value
		template<typename T, typename = std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>>
		constexpr raw_code(T v) : value(v) {}
	};

	/// serializable type that can carry error information and later reconstruct packed error
	struct BS_API box {
		int ec;
		std::string domain, message;

		box() = default;
		box(const error& er);
		box(int ec, std::string domain, std::string message) noexcept;
	};

	/// registers inherited error category automatically
	template<typename Category>
	class category : public std::error_category {
	public:
		// create instance of Category & auto-register it
		static auto self() -> Category const& {
			static const auto& self_ = make_registered_self();
			return self_;
		}

	private:
		static auto make_registered_self() -> Category const& {
			static const auto self_ = Category{};
			error::register_category(&self_);
			return self_;
		}
	};
	/// [NOTE] expects that `cat` is singleton instance
	static auto register_category(std::error_category const* cat) -> void;

private:
	// should we log error in constructor?
	enum class IsQuiet { Yes, No };

	// trigger that enables forwarding constructor
	template<typename... As>
	struct allow_pf_ctor {
		using T1 = meta::a1_t<As...>;
		static constexpr bool value = !(std::is_base_of_v<error, T1> || std::is_same_v<T1, IsQuiet>);
	};

	template<typename... As> static constexpr bool allow_pf_ctor_v = allow_pf_ctor<As...>::value;

	// test if error can be constructed with given args
	// std::is_constructible isn't applicable as we check private ctors
	template<typename... Ts>
	static constexpr auto is_constructible(int)
	-> decltype(error(std::declval<IsQuiet>(), std::declval<Ts>()...), bool{}) { return true; }

	template<typename... Ts>
	static constexpr bool is_constructible(...) { return false; }

	template<typename... Ts> static constexpr auto is_constructible_v = is_constructible<Ts...>(0);

public:
	/// code of error is stored here
	const std::error_code code;

	///////////////////////////////////////////////////////////////////////////////
	//  constructors
	//
	/// perfect forwarding ctor - construct 'non-quiet' error with Error::Happened code by default
	template<typename... Ts, typename = std::enable_if_t< allow_pf_ctor_v<Ts...> >>
	error(Ts&&... args) noexcept : error(IsQuiet::No, std::forward<Ts>(args)...) {}

	/// construct quiet error that don't get logged in constructor
	/// quiet error can be treated like operation result
	/// will construct error_code with Error::OK status by default
	template<typename... Ts>
	static auto quiet(Ts&&... args) noexcept -> std::enable_if_t<is_constructible_v<Ts...>, error> {
		return error(IsQuiet::Yes, std::forward<Ts>(args)...);
	}

	///////////////////////////////////////////////////////////////////////////////
	//  public API
	//
	/// returns error message
	auto what() const -> std::string;

	/// get error domain (category name of contained error_code)
	auto domain() const noexcept -> const char*;

	/// returns custom error message that was passed to constructor
	auto message() const noexcept -> const char*;

	/// write error to kernel log
	auto dump() const -> void;

	/// returns true if no error happened
	/// equal to !(bool)code
	auto ok() const noexcept -> bool;
	/// converison to bool returns if error happened (not ok)
	operator bool() const noexcept { return !ok(); }

	/// pack error to serializable box
	auto pack() const -> box;
	/// unpack error from box
	static auto unpack(box b) noexcept -> error;

	/// eval errors of functions sequence
	template<typename F, typename... Fs>
	static auto eval(F&& f, Fs&&... fs) -> error {
		using f_result = meta::remove_cvref_t<std::invoke_result_t<F>>;
		static const auto eval_f = [](auto&& ff) -> error {
			if constexpr(std::is_same_v<f_result, void>) {
				std::invoke(std::forward<F>(ff));
				return quiet_flag{true};
			}
			else if constexpr(is_constructible_v<f_result>)
				return std::invoke(std::forward<F>(ff));
			// allow implicit conversion chain f_result -> bool -> error (prohibited by constructors)
			else if constexpr(std::is_convertible_v<f_result, bool>) {
				return static_cast<bool>( std::invoke(std::forward<F>(ff)) );
			}
			else
				static_assert(
					meta::static_false<f_result>, "Cannot derive error from functor return type"
				);
		};

		auto er = eval_f(std::forward<F>(f));
		return er ? er : eval(std::forward<Fs>(fs)...);
	}

	// close eval recursion chain
	static inline auto eval() noexcept -> error { return true; }

	// same as `eval()` but also convert exceptions to errors
	template<typename F, typename... Fs>
	static auto eval_safe(F&& f, Fs&&... fs) noexcept {
		return eval_safe_impl(IsQuiet::No, std::forward<F>(f), std::forward<Fs>(fs)...);
	}
	// produces quiet errors
	// can be used were skipping smth in case of error is normal
	template<typename F, typename... Fs>
	static auto eval_safe_quiet(F&& f, Fs&&... fs) noexcept {
		return eval_safe_impl(IsQuiet::Yes, std::forward<F>(f), std::forward<Fs>(fs)...);
	}

	/// enable stream printing facility
	friend BS_API std::ostream& operator <<(std::ostream& os, const error& er);

	/// make formatted string representation of error
	friend BS_API std::string to_string(const error& er);

private:
	/// optional runtime error that carries message
	std::optional<std::runtime_error> info;

	/// construct from error code solely
	error(IsQuiet, std::error_code = Error::Undefined) noexcept;
	/// construct from message and error code
	error(IsQuiet, std::string_view message, std::error_code = Error::Undefined) noexcept;
	/// construct from message, int code and possible registered category name
	error(IsQuiet, std::string_view message, int err_code, std::string_view cat_name = {}) noexcept;
	/// construct from system_error that already carries error code
	error(IsQuiet, const std::system_error& er) noexcept;
	/// construct from int error code and possible registered category name
	error(IsQuiet, raw_code err_code, std::string_view cat_name = {}) noexcept;
	/// construct quiet error depending on flag: true -> Error::OK, false -> Error::Happened
	error(IsQuiet, quiet_flag state) noexcept;
	/// unpacking error from box is always quiet
	error(IsQuiet, box b) noexcept;

	/// construct from variant type that can carry error/box or result_or_err/result_or_errbox
	template<
		typename V,
		typename = std::enable_if_t<meta::is_variant_v<V> || tl::detail::is_expected<V>::value>
	>
	error(IsQuiet, V&& v) noexcept : error([&]() -> error {
		if constexpr(meta::is_variant_v<V>) {
			constexpr auto ei = meta::alternative_index<V, error>();
			if constexpr(ei >= 0) {
				if(v.index() == std::size_t{ei})
					return std::get<std::size_t{ei}>(std::forward<V>(v));
				else
					return {IsQuiet::Yes, true};
			}
			else {
				constexpr auto eib = meta::alternative_index<V, box>();
				if constexpr(eib >= 0) {
					if(v.index() == std::size_t{eib})
						return std::get<std::size_t{eib}>(std::forward<V>(v));
					else
						return {IsQuiet::Yes, true};
				}
				else {
					static_assert(eib >= 0, "Passed variant is missing `error` or `error::box` alternative");
					return {};
				}
			}
		}
		else {
			using E = typename meta::remove_cvref_t<V>::error_type;
			static_assert(
				std::is_same_v<E, error> || std::is_same_v<E, error::box>,
				"Passed expected must contain `error` or `error::box` as second type"
			);
			// if v is in expected state, replace it with unexpected quiet OK & return error
			if(v)
				return {IsQuiet::Yes, true};
			else
				return std::forward<V>(v).error();
		}
	}()) {}

	template<typename F, typename... Fs>
	static auto eval_safe_impl(IsQuiet q, F&& f, Fs&&... fs) noexcept -> error {
		try {
			return eval(std::forward<F>(f), std::forward<Fs>(fs)...);
		}
		catch(error& e) { return std::move(e); }
		catch(const std::system_error& e) { return error(e); }
		catch(const std::exception& e) { return error(q, e.what()); }
		catch(...) { return error(q, Error::Happened); }
	}
};

/// produces quiet error from given params
template<typename... Args>
inline auto success(Args&&... args) noexcept -> error {
	return error::quiet(std::forward<Args>(args)...);
}

/// correct (no error) result indicator
inline constexpr auto perfect = error::quiet_flag{true};
/// failed result indicator
inline constexpr auto quiet_fail = error::quiet_flag{false};

/// carries result (of type T) OR error
template<class T> using result_or_err = tl::expected<T, error>;
template<class T> using result_or_errbox = tl::expected<T, error::box>;

/// denote unexpected error
template<typename... Args>
inline auto unexpected_err(Args&&... args) noexcept {
	return tl::make_unexpected(error(std::forward<Args>(args)...));
}

template<typename... Args>
inline auto unexpected_err_quiet(Args&&... args) noexcept {
	return tl::make_unexpected(error::quiet(std::forward<Args>(args)...));
}

NAMESPACE_END(blue_sky)
