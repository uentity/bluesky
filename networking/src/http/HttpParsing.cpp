#include "pch.h"


#include <HttpRequest.h>
#include <HttpResponse.h>
#include <HttpParsing.h>


using namespace std;
using namespace blue_sky;
using namespace blue_sky::http_library;

namespace
{

bool is_char(char c)
{
	return c >= 0 && c <= 127;
}

bool is_control(char c)
{
	return c >= 0 && c <= 31 || c == 127;
}

bool is_special(char c)
{
	switch (c)
	{
		case '(': case ')': case '<': case '>': case '@':
		case ',': case ';': case ':': case '\\': case '"':
		case '/': case '[': case ']': case '?': case '=':
		case '{': case '}': case ' ': case '\t':
			return true;
		default:
			return false;
	}
}

bool is_digit(char c)
{
	return c >= '0' && c <= '9';
}

bool is_normal_char(char c)
{
	return is_char(c) && !is_control(c) && !is_special(c);
}

bool is_whitespace(char c)
{
	return c == ' ' || c == '\t';
}

bool match_string(istream & stream, string const& pattern)
{
	if (!stream.good())
		return false;

	bool result = true;
	string::size_type i;
	for (i = 0; i < pattern.length(); ++i)
	{	
		if (!stream.eof()) {
			if (stream.peek() != pattern[i]) {
				result = false;				
				break;
			} else {
				stream.get();
			}
		} else {
			return false;
		}
	}	
	if (!result) {
		while (i > 0) {
			stream.putback(pattern[i]);			
			--i;
		}
	}
	return result;
}

bool match_CRLF(istream & stream)
{	
	if (! stream.eof() && stream.peek() == '\r')
	{
		stream.get();
		if (!stream.eof() && stream.peek() == '\n'){
			stream.get();
			return true;
		} else {
			stream.unget();
			return false;
		}			
	} else {
		return false;
	}
}

bool match_method (istream & stream, string & method) {
	method.clear();	
	while (!stream.eof())
	{		
		if (is_normal_char(stream.peek()))
			method.push_back(stream.get());
		else {			
			break;
		}
	}
	if (stream.eof())
		return false;
	return method.length() > 0;
}

bool match_uri(istream & stream, string & uri)
{
	uri.clear();
	for(;;)
	{
		if (stream.eof())
			return false;	
		if (!is_control(stream.peek()) 
			&& stream.peek() != ' ')
		{
			uri.push_back(stream.get());
		} else {			
			return uri.length() > 0;
		}
	}	
}

bool match_digits(istream & stream, string & result)
{
	for(;;) {
		if (stream.eof())
			return false;
		else if (is_digit(stream.peek()))
			result.push_back(stream.get());
		else
			return result.length() > 0;
	}		
}

bool match_version(istream & stream, string & version)
{
	string temp1, temp2;
	if (!match_string(stream, "HTTP/"))
		return false;
	if (!match_digits(stream, temp1))
		return false;
	if (!match_string(stream, "."))
		return false;
	if (!match_digits(stream, temp2))
		return false;
	version = temp1 + "." + temp2;
	return true;
}

bool match_status_code(istream & stream, int & status_code)
{
	string digits;
	match_digits(stream, digits);
	if (digits.length() != 3)
		return false;
	status_code = (digits[0] - '0') * 100 + (digits[1] - '0') * 10 + (digits[2] - '0');
	return true;
}

bool match_reason_phrase(istream & stream, string & reason_phrase)
{
	reason_phrase.clear();
	for(;;)
	{
		if (match_CRLF(stream))
			return true;
		if (stream.eof())
			return false;
		reason_phrase.push_back(stream.get());
	}
}

bool match_request_line(istream & stream, HttpRequest & request)
{
	if (!match_method(stream, request.method))
		return false;
	if (!match_string(stream, " "))
		return false;
	if (!match_uri(stream, request.uri))
		return false;
	if (!match_string(stream, " "))
		return false;
	if (!match_version(stream, request.http_version))
		return false;
	if (!match_string(stream, "\r\n"))
		return false;
	return true;
}

bool match_response_line(istream & stream, HttpResponse & response)
{
	if (!match_version(stream, response.http_version))
		return false;
	if (!match_string(stream, " "))
		return false;
	if (!match_status_code(stream, response.status_code))
		return false;
	if (!match_string(stream, " "))
		return false;
	if (!match_reason_phrase(stream, response.reason_phrase))
		return false;
	return true;
}

bool match_header_name(istream & stream, HttpHeader & header)
{	
	if (!stream.eof() && is_normal_char(stream.peek()))
	{
		header.name = stream.get();
		for(;;) {
			if (stream.eof())
				return false;			
			if (match_string(stream, ":")) {				
				return true;
			} else if (is_normal_char(stream.peek()))
				header.name.push_back(stream.get());
			else
				return false;
		}		
	} else {		
		return false;
	}
}

bool skip_whitespace(istream & stream)
{	
	if (stream.eof() || !is_whitespace(stream.peek()))
		return false;
	else {
		while (!stream.eof() && is_whitespace(stream.peek()) )
			stream.get();
		return true;
	}
}

bool match_header_value(istream & stream, HttpHeader & header)
{
	if (!skip_whitespace(stream))
		return false;	
	header.value.clear();
	for(;;) {		
		if (stream.eof())
			return false;
		if (match_CRLF(stream))	{			
			if (!skip_whitespace(stream))
				return true;							
		} else if (!is_control(stream.peek())) {
			header.value.push_back(stream.get());
		} else {
			return false;
		}
	}	
}

bool match_header(istream & stream, HttpHeaderCollection & headers)
{
	HttpHeader header;
	if (!match_header_name(stream, header))
		return false;
	if (!match_header_value(stream, header))
		return false;
	headers.add(header);
	return true;
}

bool match_headers(istream & stream, HttpHeaderCollection & headers)
{	
	for (;;)
	{
		if (match_CRLF(stream))
			return true;
		else if (!match_header(stream, headers))
			return false;
	}	
}

}

istream & blue_sky::http_library::parse_http_request(HttpRequest & request, istream & stream, bool & success)
{	
	if (!match_request_line(stream, request) || !match_headers(stream, request.headers))
	{
		success = false;
	} else {
		success = true;
	}
	return stream;
}



istream & blue_sky::http_library::parse_http_response(HttpResponse & response, istream & stream, bool & success)
{
	if (!match_response_line(stream, response) || !match_headers(stream, response.headers))
	{
		success = false;
	} else {
		success = true;
	}
	return stream;
}

int hex2int(char c)
{
	switch (toupper(c))
	{
	case '0': return 0;
	case '1': return 1;
	case '2': return 2;
	case '3': return 3;
	case '4': return 4;
	case '5': return 5;
	case '6': return 6;
	case '7': return 7;
	case '8': return 8;
	case '9': return 9;
	case 'A': return 10;
	case 'B': return 11;
	case 'C': return 12;
	case 'D': return 13;
	case 'E': return 14;
	case 'F': return 15;
	default:
		return -1;
	}
}

std::string blue_sky::http_library::decode_uri(string const& uri)
{
	typedef string::const_iterator CIT;
	std::string result;
	result.reserve(uri.length());
	for (CIT it = uri.begin(), end_it = uri.end(); it != end_it; ++it)
	{
		if (*it == '%' && end_it - it > 2)
		{
			char h = *(++it);
			char l = *(++it);
			int ih = hex2int(h);
			int il = hex2int(l);
			if (ih < 0 || il < 0)
			{
				result.push_back('%');
				result.push_back(h);
				result.push_back(l);
			} else {
				result.push_back((char) (ih*16 + il));
			}
		} else {
			result.push_back(*it);
		}
	}
	return result;
}