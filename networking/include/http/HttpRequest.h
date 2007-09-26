#ifndef __REQUEST_H__F70CFF5C_2989_4378_BEB9_B046E67D7026_
#define __REQUEST_H__F70CFF5C_2989_4378_BEB9_B046E67D7026_

#include <string>
#include <vector>

#include "HttpHeader.h"

namespace blue_sky
{
	namespace http_library
	{
		struct HttpRequest
		{
			std::string method;
			std::string uri;
			std::string http_version;
			HttpHeaderCollection headers;
			HttpRequest()
				: http_version("1.1")
			{}
		};
	}
}
#endif //__REQUEST_H__F70CFF5C_2989_4378_BEB9_B046E67D7026_