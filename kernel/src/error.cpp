/// @file
/// @author uentity
/// @date 18.08.2016	
/// @brief Contains implimentations of BlueSky exceptions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include <bs/misc.h>
#include <bs/log.h>

#include <cstring>
#include <iostream>

#ifdef _WIN32
#include "windows.h"
#elif defined(UNIX)
#include <errno.h>
#include <string.h>
#endif

#ifdef BS_EXCEPTION_COLLECT_BACKTRACE
#include <bs/kernel_tools.h>
#endif

using namespace std;

NAMESPACE_BEGIN(blue_sky)
/*-----------------------------------------------------------------------------
 *  hidden details
 *-----------------------------------------------------------------------------*/
namespace {

#ifdef BS_EXCEPTION_COLLECT_BACKTRACE
std::string collect_backtrace() {
	return kernel_tools::get_backtrace(128);
}
#else
std::string collect_backtrace() {
	return "";
}
#endif

// not throwing error message formatter
auto format_errmsg = [](const std::string ec_message, auto custom_message = "") {
	return ec_message.size() ? ec_message + ": " + custom_message : custom_message;
};

} // hidden namespace

/*-----------------------------------------------------------------------------
 *  Default error category for generic exception
 *-----------------------------------------------------------------------------*/
std::error_code make_error_code(Error e) {
	// implement error categiry for default error code
	static const struct : std::error_category {
		const char* name() const noexcept override {
			return "blue-sky";
		}

		std::string message(int ec) const override {
			// in any case we should just substitute custom error message
				return "";
		}
	} default_category;

	return { static_cast<int>(e), default_category };
}

/*-----------------------------------------------------------------------------
 *  error implementation
 *-----------------------------------------------------------------------------*/
error::error(const char* message, const std::error_code ec, bool quiet)
	: runtime_error(format_errmsg(ec.message(), message)), code(ec)
{
	if(!quiet) dump();
}

error::error(const std::string& message, const std::error_code ec, bool quiet)
	: runtime_error(format_errmsg(ec.message(), message)), code(ec)
{
	if(!quiet) dump();
}

error::error(const char* message, const std::error_code ec)
	: error(message, std::move(ec), false)
{}

error::error(const std::string& message, const std::error_code ec)
	: error(message, std::move(ec), false)
{}

error::error(const std::error_code ec)
	: error("", std::move(ec), false)
{}

// copy ctor
error::error(const error& rhs) noexcept
	: runtime_error(rhs), code(rhs.code)
{}

//! error category
const char* error::domain() const noexcept {
	return code.category().name();
}

std::string error::to_string() const {
	return fmt::format("[{}] [{}] {}", domain(), code.value(), what());
}

void error::dump() const {
	//const auto msg = fmt::format("[{}] [{}] {}", domain(), code.value(), what());
	if(code)
		bserr() << log::E(to_string()) << log::end;
	else
		bsout() << log::I(to_string()) << log::end;
}

// exception printing
BS_API std::ostream& operator <<(std::ostream& os, const error& ec) {
	return os << ec.to_string();
}


NAMESPACE_END(blue_sky)

