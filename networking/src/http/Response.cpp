#include "pch.h"

#include <Response.h>

#include <HttpChunkedBuffer.h>

#include <HttpResponse.h>

#include <HttpParsing.h>

#include <HttpFormatting.h>

#include <networking/ISerializable.h>

#include <Http.h>

using namespace std;
using namespace boost;
using namespace blue_sky;
using namespace blue_sky::networking;
using namespace blue_sky::http_library;

class Response::Impl
{
	ContextPtr fContext;
public:
	Impl(ContextPtr context)
		: fContext(context)
	{
	}
	
	int get_buffer_size()
	{
		return fContext->buffer_size;
	}

	kernel & get_kernel()
	{
		return * (fContext->kernel);
	}

};

Response::Response(ContextPtr context)
: pimpl(new Impl(context))
{
}

ostream & Response::write_to_stream(std::ostream & stream) const
{
	HttpResponse rsp;
	rsp.status_code = Status_code;
	rsp.reason_phrase = Reason_Phrase;
	
	if (Body)
	{
		const ISerializable * body = dynamic_cast<const ISerializable *>(Body.get());
		if (body)
		{		
			string content_type = 
				string("X-bluesky-") + 
				(std::string) Body->bs_resolve_type();
			rsp.headers.add(http::CONTENT_TYPE, content_type);
			rsp.headers.add(http::TRANSFER_ENCODING, "Chunked");
			rsp.headers.add(http::ETAG, string("\"") + ETag + "\"");
			std::ostrstream str; 
			str << Last_Modified << std::ends;
			rsp.headers.add(http::LAST_MODIFIED, str.str());
			//std::cout << "ETag" << " << " << ETag << std::endl;

			format_http_response(rsp, stream);
			HttpChunkedBuffer buffer(0, &stream, pimpl->get_buffer_size());
			std::ostream str2(&buffer);
			body->serialize(str2);
			str2.flush();
			buffer.write_last_chunk();
			stream.flush();
		}
		else
		{
			throw bs_exception("Request", 
				"Body isn't serializable.");
		}
	}
	else
	{
		format_http_response(rsp, stream);
		stream.flush();
	}
	return stream;
}

istream & Response::read_from_stream(std::istream & stream)
{
	HttpResponse rsp;
	bool success;
	parse_http_response(rsp, stream, success);

	if (!success)
	{
		stream.setstate(ios::badbit);
		return stream;
	}

	Status_code = rsp.status_code;
	Reason_Phrase = rsp.reason_phrase;

	if (Status_code != http::STATUS_204_No_Content &&
		rsp.headers.contains(http::CONTENT_TYPE))
	{
		string x_type = rsp.headers.get_value(http::CONTENT_TYPE);
		static const std::string prefix = "X-bluesky-";
		if (x_type.substr(0, prefix.length()) != prefix)		
		{
			stream.setstate(ios::badbit);
			Body = sp_obj();			
			return stream;
		}

		std::string object_type = x_type.substr(prefix.length());

		Body = pimpl->get_kernel().create_object(object_type);
		bs_locker<objbase> locked = Body->lock();
		ISerializable * body = dynamic_cast<ISerializable*>(&*locked);	
		if (body)
		{
			if (rsp.headers.get_value(http::TRANSFER_ENCODING) == "Chunked")
			{			
				HttpChunkedBuffer buffer(&stream, 0, pimpl->get_buffer_size());
				std::istream str2(&buffer);
				body->deserialize(str2);
				while (!str2.eof())
					str2.get();
			}
			else
			{
				body->deserialize(stream);
			}
		}
		else
		{
			stream.setstate(ios::badbit);
			Body = sp_obj();
		}

		if (rsp.headers.contains(http::ETAG))
		{
			std::string s = rsp.headers.get_value(http::ETAG);
			string::iterator it = s.begin();
			string::iterator it2 = s.end();

			if (*it == '\"')
				++it;
			if (*(it2 - 1) == '\"')
				--it2;

			ETag = string(it, it2);
			//std::cout << "ETag" << " >> " << ETag << std::endl;
				
		}

		if (rsp.headers.contains(http::LAST_MODIFIED))
		{
			std::istrstream str(rsp.headers.get_value(http::LAST_MODIFIED).c_str());
			str >> Last_Modified;
		}
	}

	return stream;
}