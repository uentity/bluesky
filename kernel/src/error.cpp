/// @file
/// @author uentity
/// @date 18.08.2016	
/// @brief Contains implimentations of BlueSky exceptions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include <bs/tree/errors.h>
#include <bs/misc.h>
#include <bs/log.h>
#include <bs/kernel_tools.h>

#include <cstring>
#include <iostream>

using namespace std;

NAMESPACE_BEGIN(blue_sky)
/*-----------------------------------------------------------------------------
 *  hidden details
 *-----------------------------------------------------------------------------*/
namespace {

// not throwing error message formatter
inline std::string format_errmsg(const std::string ec_message, const std::string custom_message) {
	return ec_message.size() ?
		(custom_message.size() ? ec_message + ": " + custom_message : ec_message)
		: custom_message;
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
error::error(IsQuiet quiet, const std::string message, const std::error_code ec)
	: runtime_error(format_errmsg(ec.message(), std::move(message))),
	  code(ec == Error::Undefined ? (quiet == IsQuiet::Yes ? Error::OK : Error::Happened) : std::move(ec))
{
	if(quiet == IsQuiet::No) dump();
}

error::error(IsQuiet quiet, const std::error_code ec)
	: runtime_error(ec.message()),
	  code(ec == Error::Undefined ? (quiet == IsQuiet::Yes ? Error::OK : Error::Happened) : std::move(ec))
{
	if(quiet == IsQuiet::No) dump();
}

error::error(IsQuiet quiet, int ec)
	: runtime_error(""), code(static_cast<Error>(ec))
{
	//BSOUT << "error: from int!" << log::end;
	if(quiet == IsQuiet::No) dump();
}

// copy & move ctors are default
error::error(const error& rhs) noexcept = default;
error::error(error&& rhs) noexcept = default;

const char* error::domain() const noexcept {
	return code.category().name();
}

std::string error::to_string() const {
	std::string s = fmt::format("[{}] [{}] {}", domain(), code.value(), what());
#if defined(_DEBUG) && !defined(_MSC_VER)
	if(!ok()) s += kernel_tools::get_backtrace(20, 4);
#endif
	return s;
}

void error::dump() const {
	//const auto msg = fmt::format("[{}] [{}] {}", domain(), code.value(), what());
	if(code)
		bserr() << log::E(to_string()) << log::end;
	else
		bsout() << log::I(to_string()) << log::end;
}

bool error::ok() const {
	static const auto tree_extra_ok = tree::make_error_code(tree::Error::OKOK);

	return !(bool)code || (code == tree_extra_ok);
}

// error printing
BS_API std::ostream& operator <<(std::ostream& os, const error& ec) {
	return os << ec.to_string();
}

NAMESPACE_END(blue_sky)

