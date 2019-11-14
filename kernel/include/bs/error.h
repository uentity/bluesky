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
#include <exception>
#include <system_error>
#include <ostream>

#define BS_REGISTER_ERROR_ENUM(E) \
namespace std { template<> struct is_error_code_enum< E > : true_type {}; }

#define EVAL if(auto er = ::blue_sky::error::eval(
#define EVAL_SAFE if(auto er = ::blue_sky::error::eval_safe(
#define RETURN_EVAL_ERR )) return er;

#define SCOPE_EVAL if(auto er = ::blue_sky::error::eval([&] {
#define SCOPE_EVAL_SAFE if(auto er = ::blue_sky::error::eval_safe([&] {
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
class BS_API error : public std::runtime_error {
public:
	// indicates that no error happened
	struct success_tag {};

	/// registers inherited error category automatically
	template<typename Category>
	class category : public std::error_category {
	public:
		// create instance of Category & auto-register it
		static auto self() -> Category const& {
			const auto& self_ = []() -> const auto& {
				static const auto self_ = Category{};
				error::register_category(&self_);
				return self_;
			}();
			return self_;
		}
	};
	/// [NOTE] expects that `cat` is singleton instance
	static auto register_category(std::error_category const* cat) -> void;

	/// serializable type that can carry error information and later reconstruct packed error
	struct box {
		int ec;
		std::string message, domain;

		box() = default;
		box(const error& er);
		box(int ec, std::string message, std::string domain);
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
	error(Ts&&... args) : error(IsQuiet::No, std::forward<Ts>(args)...) {}

	/// copy & move ctors - must not throw
	error(const error& rhs) noexcept;
	error(error&& rhs) noexcept;
	~error() noexcept {}

	/// construct quiet error with OK status
	error(success_tag);

	/// construct quiet error that don't get logged in constructor
	/// quiet error can be treated like operation result
	/// will construct error_code with Error::OK status by default
	template<typename... Ts>
	static error quiet(Ts&&... args) {
		return error(IsQuiet::Yes, std::forward<Ts>(args)...);
	}

	/// returns error message
	using std::runtime_error::what;

	/// get error domain (category name of contained error_code)
	const char* domain() const noexcept;

	/// returns custom error message that was passed to constructor
	const char* message() const noexcept;

	/// write error to kernel log
	void dump() const;

	/// returns true if no error happened
	/// equal to !(bool)code
	bool ok() const;
	/// converison to bool returns if error happened (not ok)
	operator bool() const {
		return !ok();
	}

	/// pack error to serializable box
	auto pack() const -> box;
	/// unpack error from box
	static auto unpack(box b) -> error;

	/// eval errors of functions sequence
	static inline auto eval() -> error {
		return success_tag{};
	}

	template<typename F, typename... Fs>
	static auto eval(F&& f, Fs&&... fs) -> error {
		// detect if f return void, result_or_err or simply error
		// otherwise raise static assertion
		using f_result = std::invoke_result_t<F>;
		const auto eval_f = [](auto&& ff) -> error {
			if constexpr(std::is_same_v<f_result, error>)
				return std::invoke<F>(std::forward<F>(ff));
			else if constexpr(std::is_same_v<f_result, void>) {
				std::invoke<F>(std::forward<F>(ff));
				return success_tag{};
			}
			else if constexpr(tl::detail::is_expected<f_result>::value) {
				static_assert(
					std::is_same_v<typename f_result::error_type, error>,
					"Returned expected must contain error as second type"
				);
				// [NOTE] x.value is ignored!
				auto x = std::invoke<F>(std::forward<F>(ff));
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
	static auto eval_safe(F&& f, Fs&&... fs) noexcept -> error {
		try {
			return eval(std::forward<F>(f), std::forward<Fs>(fs)...);
		}
		catch(const std::system_error& e) { return {e}; }
		catch(const std::exception& e) { return {e.what()}; }
		catch(...) { return {}; }
	}

	/// enable stream printing facility
	friend BS_API std::ostream& operator <<(std::ostream& os, const error& er);

	/// make formatted string representation of error
	friend BS_API std::string to_string(const error& er);

private:
	/// construct from message and error code
	explicit error(IsQuiet, std::string message, std::error_code = Error::Undefined);
	/// construct from error code solely
	explicit error(IsQuiet, std::error_code = Error::Undefined);
	/// construct from message, int code and possible registered category name
	explicit error(IsQuiet, std::string message, int err_code, std::string_view cat_name = "");
	/// construct from int error code and possible registered category name
	explicit error(IsQuiet, int err_code, std::string_view cat_name = "");
	// construct from system_error that already carries error code
	explicit error(IsQuiet, const std::system_error& er);
	/// unpacking error from box is always quiet
	explicit error(IsQuiet, box b);
};

/// produces quiet error from given params
template<typename... Args>
inline auto success(Args&&... args) -> error {
	return error::quiet(std::forward<Args>(args)...);
}

/// signle value indicating correct (no error) result
inline constexpr auto perfect = error::success_tag{};

/// carries result (of type T) OR error
template<class T> using result_or_err = tl::expected<T, error>;
template<class T> using result_or_errbox = tl::expected<T, error::box>;

NAMESPACE_END(blue_sky)

