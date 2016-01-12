/// @file
/// @author NikonovMA a ka no_NaMe <__no_name__@rambler.ru>, uentity, Sergey Miryanov
/// @date 12.01.2016
/// @brief Contains BlueSky exception class declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_EXCEPTION_H_
#define BS_EXCEPTION_H_

#include "bs_common.h"
#include <exception>

#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
#include <boost/format.hpp>
#endif

//! \defgroup journaling journaling - blue-sky journaling classes

namespace blue_sky 
{

   /*!
     \class bs_exception
     \ingroup journaling
     \brief blue-sky base exceptions class
    */
  class BS_API bs_exception : public std::exception
  {
  public:
    /**
     * \param who - who raise this exception
     * \param message - exception message
     * */
    bs_exception(const std::string & who, const std::string & message);
    
#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
    /**
     * \param who - who (and mostly where) raise exception
     * \param message - exception message (boost::format class instance)
     * */
    bs_exception (const std::string &who, const boost::format &message);
#endif

    //! \brief Destructor.
    virtual ~bs_exception() throw()
    {
    }
    
    //! \brief says a bit more info about exception
    virtual const char* 
    what() const throw();

    //! \brief says who raise this exception
    virtual const char* 
    who() const;
    
  protected:
    std::string who_;             //!< Who message
    std::string what_;            //!< what message
    blue_sky::error_code m_err_;  //!< error code of exception

  private:
    static void *
    operator new (size_t nbytes);
  };

  /**
   * \brief blue-sky kernel exception
   * */
  class BS_API bs_kernel_exception : public bs_exception
  {
  public:
    /**
     * \param who - who raise exception
     * \param ec - error code
     * \param message - exception message
     * */
    bs_kernel_exception (const std::string &who, error_code ec, const std::string &message);

#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
    /**
     * \param who - who raise exception
     * \param ec - error code
     * \param message - exception message
     * */
    bs_kernel_exception (const std::string &who, error_code ec, const boost::format &message);
#endif
  };

  /**
   * \brief blue-sky system exception
   * */
  class BS_API bs_system_exception : public bs_exception
  {
  public:
    /**
     * \param who - who raise exception
     * \param ec - error code
     * \param message - exception message
     * */
    bs_system_exception (const std::string &who, error_code ec, const std::string &message);

#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
    /**
     * \param who - who raise exception
     * \param ec - error code
     * \param message - exception message
     * */
    bs_system_exception (const std::string &who, error_code ec, const boost::format &message);
#endif
  };
  
  /**
   * \brief custom exception for dynamic library errors
   * */
  class BS_API bs_dynamic_lib_exception : public bs_exception
  {
  public:
    /**
     * \param who - who raise exception
     * */
    bs_dynamic_lib_exception (const std::string &who);

#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
    /**
     * \param who - who raise exception
     * */
    explicit bs_dynamic_lib_exception (const boost::format &who);
#endif
  };

}

#endif // #ifndef BS_EXCEPTION_H_

