/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Contains implimentations of blue-sky exception class
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "bs_exception.h"
//#include "bs_log.h"
#include "bs_report.h"
#include "bs_misc.h"
#include "bs_link.h"
#ifndef UNIX
  #include "windows.h"
#endif

#ifdef UNIX
  #include <errno.h>
  #include <string.h>
#endif

#ifdef BS_EXCEPTION_COLLECT_BACKTRACE
#include "bs_kernel_tools.h"
#endif

using namespace std;

namespace blue_sky
{
  namespace detail 
  {
#ifdef BS_EXCEPTION_COLLECT_BACKTRACE
    std::string 
    collect_backtrace ()
    {
      return kernel_tools::get_backtrace (128);
    }
#endif
  } // namespace detail


  bs_exception::bs_exception(const std::string &who, const std::string &message)
  : who_(who), 
  what_ (who_ + ": " + message),
  m_err_(user_defined)
  {
#ifdef BS_EXCEPTION_COLLECT_BACKTRACE
    what_ += detail::collect_backtrace ();
#endif
  }

#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
  bs_exception::bs_exception (const std::string &who, const boost::format &message)
  : who_ (who),
  what_ (who_ + ": " + message.str ()),
  m_err_ (user_defined)
  {
#ifdef BS_EXCEPTION_COLLECT_BACKTRACE
    what_ += detail::collect_backtrace ();
#endif
  }
#endif

  //! \return exception message string
  const char* bs_exception::what() const throw()
  {
    return what_.c_str();
  }

  //! \return who provoked exception
  const char* bs_exception::who() const
  {
    return who_.c_str();
  }

  bs_kernel_exception::bs_kernel_exception (const std::string &who, error_code ec, const std::string &what)
  : bs_exception (who, what + ". BlueSky error: " + bs_error_message (ec))
  {
    m_err_ = ec;
  }
#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
  bs_kernel_exception::bs_kernel_exception (const std::string &who, error_code ec, const boost::format &message)
  : bs_exception (who, message.str () + ". BlueSky error: " + bs_error_message (ec))
  {
    m_err_ = ec;
  }
#endif

  bs_system_exception::bs_system_exception (const std::string &who, error_code ec, const std::string &what)
  : bs_exception (who, what + ". System error: " + system_message (ec))
  {
    m_err_ = ec;
  }
#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
  bs_system_exception::bs_system_exception (const std::string &who, error_code ec, const boost::format &message)
  : bs_exception (who, message.str () + ". System error: " + system_message (ec))
  {
    m_err_ = ec;
  }
#endif

  /**
   * \brief bs_dynamic_lib_exception
   * */
  bs_dynamic_lib_exception::bs_dynamic_lib_exception (const std::string &who)
  : bs_exception (who, "System error: " + dynamic_lib_error_message ())
  {
    m_err_ = user_defined;
  }
#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
  bs_dynamic_lib_exception::bs_dynamic_lib_exception (const boost::format &who)
  : bs_exception (who.str (), "System error: " + dynamic_lib_error_message ())
  {
    m_err_ = user_defined;
  }
#endif
}

