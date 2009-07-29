#include "pch.h"

#include "HttpFormatting.h"
#include "HttpRequest.h"
#include "HttpResponse.h"

using namespace std;
using namespace blue_sky::http_library;

#define CRLF "\r\n"

namespace {
ostream & format_headers(const HttpHeaderCollection & headers, ostream & stream)
{
	struct Func
	{
		ostream & stream_;
	public:
		Func(ostream & stream) : stream_(stream){}
		void operator()(const string & name, string const& value)
		{
			stream_ << name << ": " << value << CRLF;
		}
	} f (stream);	
	headers.iterate(f);
	stream << CRLF;
	return stream;
}
}

ostream & blue_sky::http_library::format_http_request(HttpRequest const& request, ostream & stream)
{
	
	return format_headers(request.headers, 
		stream << request.method << " " << request.uri << " HTTP/" << request.http_version << CRLF);		
	
}

ostream & blue_sky::http_library::format_http_response(HttpResponse const& response, ostream & stream)
{	
	return format_headers(response.headers, 
		stream << "HTTP/" << response.http_version <<  " " << response.status_code << " " << response.reason_phrase << CRLF);	
}