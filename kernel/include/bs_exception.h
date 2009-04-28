// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

/*!
  \file bs_exception.h
  \brief Contains blue-sky exception class
  \author NikonovMA a ka no_NaMe <__no_name__@rambler.ru>
 */
#ifndef __BS_EXCEPTION_H
#define __BS_EXCEPTION_H

#include "bs_common.h"
#include <exception>

//! \defgroup journaling journaling - blue-sky journaling classes

namespace blue_sky 
{

	 /*!
		 \class bs_exception
		 \ingroup journaling
		 \brief blue-sky exceptions class
		*/
	class BS_API bs_exception : public std::exception
	{
		public:
		 //! \brief Constructor
			bs_exception(const std::string & who, //!< Who provoked exception
				const std::string & message //!< double message to user
									 );
		 //! \brief Constructor with more parameters
			bs_exception(const char* who, //!< Who provoked exception
									 const error_code ec, //!< blue_sky error_code of error
									 const char* message, //!< double message to user
									 bool system_ = false, //!< Is it system error
									 const char* param1 = "", //!< double message
									 const char* param2 = "" //!< double message
									 );
			/*			bs_exception(const bs_exception &e) 
			{
				//std::cout << "this string" << std::endl;
			}							
			*/
			//! \brief Destructor.
			virtual ~bs_exception() throw()
			{
				
			};	// no exceptions cought (all exc - unexpected)
			
			//! \brief What-method of exception
			virtual const char* what() const throw();
			//! \brief Who-method of exception
			const char* who() const;
			
			//set_terminate(TERMINTE_FUN(what()))
		private:
			std::string who_; //!< Who message
			std::string what_; //!< what message
			//class impl;
			//impl impl_;
			blue_sky::error_code m_err_; //!< error code of exception
	};
	/*
	void throw_exception(const std::string& who, const error_code ec, const std::string& message, 
		const std::string& param1 = "", const std::string& param2 = "")
		{
			throw bs_exception("smb",blue_sky::out_of_range,"some msg","param1","param2");
		}*/
}

#endif // __BS_EXCEPTION_H

