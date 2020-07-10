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
	// indicates that no error happened
	struct success_tag {};

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

	/// serializable type that can carry error information and later reconstruct packed error
	struct BS_API box {
		int ec;
		std::string domain, message;

		box() = default;
		box(const error& er);
		box(int ec, std::string domain, std::string message) noexcept;
	};

private:
	// should we log error in constructor?
	enum class IsQuiet { Yes, No };

	// helper to narrow gready nature of perfect forwarding ctor
	template<typename A1 = int, typename... As>
	struct allow_forward {
		using T1 = std::remove_cv_t<std::remove_reference_t<A1>>;
		static constexpr bool value = !std::is_same_v<T1, IsQuiet> && !std::is_same_v<T1, success_tag>
			&& !std::is_base_of_v<error, T1>;
	};

public:
	/// code of error is stored here
	const std::error_code code;

	/// perfect forwarding ctor - construct 'non-quiet' error with Error::Happened code by default
	template<
		typename... Ts,
		typename = std::enable_if_t< allow_forward<Ts...>::value >
	>
	error(Ts&&... args) noexcept : error(IsQuiet::No, std::forward<Ts>(args)...) {}
	/// construct quiet error with OK status
	error(success_tag) noexcept;

	/// copy & move ctors
	error(const error& rhs) = default;
	error(error&& rhs) = default;

	/// construct quiet error that don't get logged in constructor
	/// quiet error can be treated like operation result
	/// will construct error_code with Error::OK status by default
	template<typename... Ts>
	static auto quiet(Ts&&... args) noexcept -> error {
		return error(IsQuiet::Yes, std::forward<Ts>(args)...);
	}

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
	static inline auto eval() noexcept -> error {
		return success_tag{};
	}

	template<typename F, typename... Fs>
	static auto eval(F&& f, Fs&&... fs) -> error {
		// detect if f return void, result_or_err or simply error
		// otherwise raise static assertion
		using f_result = std::invoke_result_t<F>;
		const auto eval_f = [](auto&& ff) -> error {
			if constexpr(
				std::is_same_v<f_result, error> || std::is_same_v<f_result, success_tag> ||
				std::is_error_code_enum_v<f_result>
			)
				return std::invoke(std::forward<F>(ff));
			else if constexpr(std::is_same_v<f_result, void>) {
				std::invoke(std::forward<F>(ff));
				return success_tag{};
			}
			else if constexpr(std::is_convertible_v<f_result, bool>) {
				// convert `false` to quiet error
				return std::invoke(std::forward<F>(ff)) ?
					success_tag{} : quiet(Error::Happened);
			}
			else if constexpr(tl::detail::is_expected<f_result>::value) {
				static_assert(
					std::is_same_v<typename f_result::error_type, error>,
					"Returned expected must contain error as second type"
				);
				// [NOTE] x.value is ignored!
				auto x = std::invoke(std::forward<F>(ff));
				return x ? success_tag{} : x.error();
			}
			else
				static_assert(
					std::is_same_v<f_result, error>,
					"Cannot derive error from functor return type"
				);
		};

		auto er = eval_f(std::forward<F>(f));
		return er ? er : eval(std::forward<Fs>(fs)...);
	}

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

	/// construct from message and error code
	explicit error(IsQuiet, std::string_view message, std::error_code = Error::Undefined) noexcept;
	/// construct from error code solely
	explicit error(IsQuiet, std::error_code = Error::Undefined) noexcept;
	/// construct from message, int code and possible registered category name
	explicit error(IsQuiet, std::string_view message, int err_code, std::string_view cat_name = "") noexcept;
	/// construct from int error code and possible registered category name
	explicit error(IsQuiet, int err_code, std::string_view cat_name = "") noexcept;
	// construct from system_error that already carries error code
	explicit error(IsQuiet, const std::system_error& er) noexcept;
	/// unpacking error from box is always quiet
	explicit error(IsQuiet, box b) noexcept;

	template<typename F, typename... Fs>
	static auto eval_safe_impl(IsQuiet q, F&& f, Fs&&... fs) noexcept -> error {
		try {
			return eval(std::forward<F>(f), std::forward<Fs>(fs)...);
		}
		catch(error& e) { return std::move(e); }
		catch(const std::system_error& e) { return error{q, e}; }
		catch(const std::exception& e) { return error{q, e.what()}; }
		catch(...) { return error{q, Error::Happened}; }
	}
};

/// produces quiet error from given params
template<typename... Args>
inline auto success(Args&&... args) noexcept -> error {
	return error::quiet(std::forward<Args>(args)...);
}

/// signle value indicating correct (no error) result
inline constexpr auto perfect = error::success_tag{};

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
