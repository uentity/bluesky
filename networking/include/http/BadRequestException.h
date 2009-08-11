#ifndef __BADREQUESTEXCEPTION_H__304AB659_5ABF_4884_8A00_22BD94AB2DC3_
#define __BADREQUESTEXCEPTION_H__304AB659_5ABF_4884_8A00_22BD94AB2DC3_

#include <stdexcept>
#include <string>

namespace blue_sky
{
	namespace http_library
	{
		class BadRequestException : public std::exception
		{
			virtual const char *what( ) const
			{
				static std::string message = "Bad request.";
				return message.c_str();
			}
		};
	}
}

#endif //__BADREQUESTEXCEPTION_H__304AB659_5ABF_4884_8A00_22BD94AB2DC3_