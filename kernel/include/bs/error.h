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
#include <exception>
#include <system_error>
#include <ostream>

#define BS_REGISTER_ERROR_ENUM(E) \
namespace std { template<> struct is_error_code_enum< E > : true_type {}; }

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
	// should we log error in constructor?
	enum class IsQuiet { Yes, No };

	// helper to narrow gready nature of perfect forwarding ctor
	template<typename A1 = int, typename... As>
	struct allow_forward {
		static constexpr bool value = !std::is_same<A1, IsQuiet>::value
			&& !std::is_base_of<error, std::decay_t<A1>>::value;
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

	/// construct quiet error that don't get logged in constructor
	/// quiet error can be treated like operation result
	/// will construct error_code with Error::OK status by default
	template<typename... Ts>
	static error quiet(Ts&&... args) {
		return error(IsQuiet::Yes, std::forward<Ts>(args)...);
	}

	/// use what() from base class
	using std::runtime_error::what;

	/// get error domain (category name of contained error_code)
	const char* domain() const noexcept;

	/// make formatted string representation of error
	std::string to_string() const;

	/// write error to kernel log
	void dump() const;

	/// returns true if no error happened
	/// equal to !(bool)code
	bool ok() const;
	/// converison to bool returns if error happened (not ok)
	operator bool() const {
		return !ok();
	}

	/// enable stream printing facility
	friend BS_API std::ostream& operator <<(std::ostream& os, const error& ec);

private:
	/// construct from message and error code
	explicit error(IsQuiet, const std::string message, const std::error_code = Error::Undefined);
	/// construct from error code solely
	explicit error(IsQuiet, const std::error_code = Error::Undefined);
	/// construct from int operation result
	explicit error(IsQuiet, int err_code);
};

NAMESPACE_END(blue_sky)

