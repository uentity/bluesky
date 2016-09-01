/// @file
/// @author uentity
/// @date 18.08.2016
/// @brief Contains BlueSky exceptions declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "common.h"
#include <exception>
#include <boost/format.hpp>
#include <ostream>

namespace blue_sky {

class BS_API bs_exception : public std::exception {
public:
	/**
		* \param who - who raise this exception
		* \param message - exception message
		* */
	bs_exception(const char* message, const char* who = "", int err_code = -1);

	/**
		* \param who - who (and mostly where) raise exception
		* \param message - exception message (boost::format class instance)
		* */
	bs_exception(const boost::format& message, const char* who = "", int err_code = -1);

	// templated ctor where params can be omitted
	template< typename T, typename... Args >
	bs_exception(const T& message, Args... args)
		: bs_exception(
			message,
			bs_args::get_value("", args...),
			bs_args::get_value(-1, args...)
		)
	{}

	//! \brief virtual destructor for derived exceptions
	virtual ~bs_exception() noexcept {}

	//! \brief says a bit more info about exception
	const char* what() const noexcept override;

	//! \brief says who raise this exception
	const char* who() const noexcept;

	// access to error code
	int err_code() const noexcept {
		return err_code_;
	}

	// enable printing & loging facility
	friend std::ostream& operator <<(std::ostream& os, const bs_exception& ex);

	// write exeption to kernel error log
	void print() const;

protected:
	std::string who_;  //!< source
	std::string what_; //!< full description
	int err_code_;     //!< Error code
};

/**
* \brief BlueSky kernel exception
* */
class BS_API bs_kexception : public bs_exception {
public:
	//using bs_exception::bs_exception;
	bs_kexception(const char* message, const char* ksubsyst = "", int err_code = -1);
};

class BS_API bs_sys_exception : public bs_exception {
public:
	//using bs_exception::bs_exception;
	bs_sys_exception(int err_code, const char* who = "");
};

//  /**
//   * \brief blue-sky system exception
//   * */
//  class BS_API bs_system_exception : public bs_exception
//  {
//  public:
//    /**
//     * \param who - who raise exception
//     * \param ec - error code
//     * \param message - exception message
//     * */
//    bs_system_exception (const std::string &who, error_code ec, const std::string &message);
//
//#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
//    /**
//     * \param who - who raise exception
//     * \param ec - error code
//     * \param message - exception message
//     * */
//    bs_system_exception (const std::string &who, error_code ec, const boost::format &message);
//#endif
//  };
//  
//  /**
//   * \brief custom exception for dynamic library errors
//   * */
//  class BS_API bs_dynamic_lib_exception : public bs_exception
//  {
//  public:
//    /**
//     * \param who - who raise exception
//     * */
//    bs_dynamic_lib_exception (const std::string &who);
//
//#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
//    /**
//     * \param who - who raise exception
//     * */
//    explicit bs_dynamic_lib_exception (const boost::format &who);
//#endif
//  };

}  // eof blue_sky namespace

BS_API std::ostream& operator <<(std::ostream& os, const blue_sky::bs_exception& ex);

