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
  \file bs_exception.cpp
  \brief Contains implimentations of blue-sky exception class
  \author NikonovMA a ka no_NaMe <__no_name__@rambler.ru>
 */

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

//#define XPN_LOG_INST log::Instance()[XPN_LOG] //!< blue-sky log for error output

using namespace std;

namespace blue_sky
{

namespace {

	void bs_terminate(void)
	{
		 BSERROR << "BlueSky terminated." << bs_end;
	}

	struct err_pair
	{
		 blue_sky::error_code e;
		 const char* descr_;
	};

	err_pair ep[] =
		 {
				{wrong_path, "Wrong path"},
				{no_plugins, "No plugins found in"},
				{out_of_range, "Out of range"},
				{user_defined, "User defined error"},
				{boost_error, "Boost error"},
				{no_library, "No such library"},
				{no_type, "No type"},
				{notpermitted_operation, "Not permitted operation"}
				//{no_error, "fuck off no error"}
		 };

	const unsigned int ep_size = 7;

	string bs_error_message(blue_sky::error_code& ec)//, const bool pass = NULL)
	{
		 for(unsigned int i = 0; i < ep_size; ++i) {
				if (ep[i].e == ec)
					 return ep[i].descr_;
		 }

		 ec = blue_sky::system_error;
		 return "System error";
	}

	namespace subsidiary //! namespace subsidiary
	{
		 /*!
			 \brief Simple connector of two strings to format "str1 : str2"
			 \param twho - string1 (who)
			 \param twhat - string2 (what)
			 \return message-string formated as "str1 : str2"
			*/
		 const string connector(const string &twho, const string &twhat)
		 {
				return (twho + ": " + twhat);
		 }

		 /*!
			 \brief Another connector of strings to format "str1 : str2. Error type: str3"
			 \param twho - string1 (who)
			 \param ec - blue_sky::error_code
			 \param twhat - string2 (what)
			 \param system_ - false if error is not system, true - otherwise
			 \param param1 - double message to user
			 \param param2 - double message to user
			 \return message-string formated as "str1 : str2. Error type: str3"
			*/
		 const string connector(const string &twho, error_code ec, const string &twhat,
														bool system_ = false, const std::string& param1 = "",
														const std::string& param2 = "")
		 {
				string str;
				if (system_ || ec == blue_sky::system_error)
					 str = twho + ": " + twhat + ". System error: " + system_message(ec);
				else
					 str = twho + ": " + twhat + ". BlueSky error: " + bs_error_message(ec);

				if (param1 != "")
					 str += " \"" + param1 + "\"";
				if (param2 != "")
					 str += ", \"" + param2 + "\"";

				//str += "...";
				return str;
		 }
	}	//end of subsidiary namespace
}	//end of hidden namespace

bs_exception::bs_exception(const std::string & who, const std::string & message)
		 : who_(who), m_err_(user_defined)
  {
		 //set_terminate(bs_terminate);
		 what_ = subsidiary::connector(who, message);
		 //if (BSERROR.outputs_time())
		 //BSERROR << output_time;
		 BSERROR << what_.c_str() << bs_end;
		 //if (!BSERROR.outputs_time())
		 //BSERROR << output_time;
  }

	 bs_exception::bs_exception(const char* who, const error_code ec, const char* message,
															bool system_, const char* param1, const char* param2)
			: who_(who), m_err_(ec)
	 {
		 //set_terminate(bs_terminate);
		 /*if(param1 == "")
			 what_ = subsidiary::connector(who,ec,message);
			 else*/
			what_ = subsidiary::connector(who_, ec, string(message), system_, string(param1), string(param2));
			//if (BSERROR.outputs_time())
			//BSERROR << output_time;
			BSERROR << what_.c_str() << bs_end;
			//if (!BSERROR.outputs_time())
			//BSERROR << output_time;
	 }

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

}

