#ifndef __HTTPPARSING_H__53D4D642_01D3_4542_B349_120F05227C78_
#define __HTTPPARSING_H__53D4D642_01D3_4542_B349_120F05227C78_

#include <iosfwd>

namespace blue_sky
{
	namespace http_library
	{
		struct HttpRequest;
		struct HttpResponse;
		std::string decode_uri(const std::string & uri);
		std::istream & parse_http_request(HttpRequest & request, std::istream & stream, bool & success);
		std::istream & parse_http_response(HttpResponse & response, std::istream & stream, bool & success);
	}
}


#endif //__HTTPPARSING_H__53D4D642_01D3_4542_B349_120F05227C78_