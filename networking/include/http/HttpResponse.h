#ifndef __HTTPRESPONSE_H__1C3482A9_9B0E_47FE_B17B_96DCF881914D_
#define __HTTPRESPONSE_H__1C3482A9_9B0E_47FE_B17B_96DCF881914D_

#include <string>
#include <vector>

#include <HttpHeader.h>

namespace blue_sky
{
	namespace http_library
	{
		struct HttpResponse
		{
			std::string http_version;
			int status_code;
			std::string reason_phrase;
			HttpHeaderCollection headers;
			HttpResponse() 
				: http_version("1.1")
			{}
		};
	}
}

#endif //__HTTPRESPONSE_H__1C3482A9_9B0E_47FE_B17B_96DCF881914D_