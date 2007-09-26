#ifndef __HTTPFORMATTER_H__0F8330C6_F50C_4EAC_B34D_C6D5AF6B6CDA_
#define __HTTPFORMATTER_H__0F8330C6_F50C_4EAC_B34D_C6D5AF6B6CDA_

#include <iosfwd>

namespace blue_sky
{
	namespace http_library
	{
		struct HttpRequest;
		struct HttpResponse;
		//std::string encode_uri(std::string const& uri);
		std::ostream & format_http_request(const HttpRequest & request, std::ostream & stream);		
		std::ostream & format_http_response(const HttpResponse & response, std::ostream & stream);
	}
}

#endif //__HTTPFORMATTER_H__0F8330C6_F50C_4EAC_B34D_C6D5AF6B6CDA_