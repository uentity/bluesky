/// @file
/// @author uentity
/// @date 18.08.2016	
/// @brief Contains implimentations of BlueSky exceptions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/exception.h>
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

namespace blue_sky {
// hide details
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

} // hidden namespace


bs_exception::bs_exception(const char* message, const char* who, int err_code)
	: who_(who),
	what_((who_.empty() ? "" : who_ + ": ") + message + collect_backtrace()),
	err_code_(err_code)
{}

bs_exception::bs_exception(const boost::format &message, const char* who, int err_code)
	: bs_exception(message.str().c_str(), who, err_code)
{}

//! \return exception message string
const char* bs_exception::what() const noexcept {
	try {
		return what_.c_str();
	}
	catch(...) {
		return "";
	}
}

//! \return who provoked exception
const char* bs_exception::who() const noexcept {
	try {
		return who_.c_str();
	}
	catch(...) {
		return "";
	}
}

// exception printing
BS_API std::ostream& operator <<(std::ostream& os, const bs_exception& ex) {
	return os << "[Exception] " << ex.what();
}

void bs_exception::print() const {
	bserr() << log::E("[Exception] {}") << what() << log::end;
}

/*-----------------------------------------------------------------
 * Kernel exception impl
 *----------------------------------------------------------------*/
bs_kexception::bs_kexception(const char* message, const char* who, int err_code)
	: bs_exception(
		message,
		(std::string("kernel") + (strlen(who) > 0 ? std::string(":") + who : "")).c_str(), err_code
	)
{}

bs_kexception::bs_kexception(const boost::format& message, const char* who, int err_code)
	: bs_exception(
		message,
		(std::string("kernel") + (strlen(who) > 0 ? std::string(":") + who : "")).c_str(), err_code
	)
{}

bs_sys_exception::bs_sys_exception(int err_code, const char* who)
	: bs_exception(system_message(err_code).c_str(), who, err_code)
{}

//  bs_system_exception::bs_system_exception (const std::string &who, error_code ec, const std::string &what)
//  : bs_exception (who, what + ". System error: " + system_message (ec))
//  {
//    m_err_ = ec;
//  }
//#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
//  bs_system_exception::bs_system_exception (const std::string &who, error_code ec, const boost::format &message)
//  : bs_exception (who, message.str () + ". System error: " + system_message (ec))
//  {
//    m_err_ = ec;
//  }
//#endif
//
//  /**
//   * \brief bs_dynamic_lib_exception
//   * */
//  bs_dynamic_lib_exception::bs_dynamic_lib_exception (const std::string &who)
//  : bs_exception (who, "System error: " + dynamic_lib_error_message ())
//  {
//    m_err_ = user_defined;
//  }
//#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
//  bs_dynamic_lib_exception::bs_dynamic_lib_exception (const boost::format &who)
//  : bs_exception (who.str (), "System error: " + dynamic_lib_error_message ())
//  {
//    m_err_ = user_defined;
//  }
//#endif

} // eof blue_sky namespace

