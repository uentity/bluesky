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


NAMESPACE_BEGIN(blue_sky)

/// error code that is assigned by default if not specified explicitly
enum class Error {
	OK = 0,
	Happened = -1
};
BS_API std::error_code make_error_code(Error);

NAMESPACE_END(blue_sky)

// register default error code
BS_REGISTER_ERROR_ENUM(blue_sky::Error)

NAMESPACE_BEGIN(blue_sky)

class BS_API error : public std::runtime_error {
public:
	/// code of error is stored here
	const std::error_code code;

	/// use what() from base class
	using std::runtime_error::what;

	/// construct from message and error code
	error(const char* message, const std::error_code = Error::Happened);
	error(const std::string& message, const std::error_code = Error::Happened);
	/// construct from error code solely
	error(const std::error_code);
	/// copy ctor - must not throw
	error(const error& rhs) noexcept;
	/// virtual destructor for derived errors
	virtual ~error() noexcept {}

	/// construct quiet error that don't get logged in constructor
	/// quiet error can be treated like operation result
	/// thus, assume OK status by default
	/// from string message
	template<
		typename Msg,
		typename = std::enable_if_t< !std::is_base_of<std::error_code, std::decay_t<Msg>>::value >
	>
	static error quiet(Msg&& message, const std::error_code c = Error::OK) {
		return {std::forward<Msg>(message), std::move(c), true};
	}
	/// from error code
	static error quiet(const std::error_code c) {
		return {"", std::move(c), true};
	}

	/// get error domain
	/// forwards to error_code::category
	const char* domain() const noexcept;

	/// make formatted string representation of error
	std::string to_string() const;

	/// enable stream printing facility
	friend BS_API std::ostream& operator <<(std::ostream& os, const error& ec);

	/// write error to kernel log
	void dump() const;

private:
	error(const char* message, const std::error_code, bool quiet);
	error(const std::string& message, const std::error_code, bool quiet);
};

NAMESPACE_END(blue_sky)

