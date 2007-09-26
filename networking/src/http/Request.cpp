#include "pch.h"
#include <http/HttpChunkedBuffer.h>
#include <networking/Context.h>

#include <http/Request.h>

#include <http/HttpRequest.h>
#include <http/Http.h>

#include <Http/HttpParsing.h>
#include <http/HttpFormatting.h>

#include <networking/ISerializable.h>

using namespace std;
using namespace boost;
using namespace blue_sky;
using namespace blue_sky::networking;
using namespace blue_sky::http_library;

class Request::Impl
{
public:
	Impl(ContextPtr context)
		: fContext(context)
	{}

	ContextPtr fContext;

	int get_buffer_size()
	{
		return fContext->client_settings->buffer_size;
	}

	kernel & get_kernel()
	{
		return * (fContext->kernel);
	}

};

Request::Request(ContextPtr context)
: pimpl(new Impl(context))
{

}


std::ostream & Request::write_to_stream(std::ostream & stream) const
{
	HttpRequest req;
	switch (method)
	{
		case M_GET : req.method = "GET"; break;
		case M_PUT : req.method = "PUT"; break;
		case M_POST : req.method = "POST"; break;
		case M_DELETE : req.method = "DELETE"; break;
		case M_LOCK : req.method = "LOCK"; break;
		case M_UNLOCK : req.method = "UNLOCK"; break;
		default: throw bs_exception("Request", "Invalid method.");
	}

	req.uri = uri.str();
	
	if (Body)
	{
		const ISerializable * body = dynamic_cast<const ISerializable *>(Body.get());
		if (body)
		{		
			string content_type = string("X-bluesky-") + (std::string) Body->bs_resolve_type();
			req.headers.add(http::CONTENT_TYPE, content_type);
			req.headers.add(http::TRANSFER_ENCODING, "Chunked");

			format_http_request(req, stream);

			HttpChunkedBuffer buffer(0, &stream, pimpl->get_buffer_size());			
			std::ostream str2 (&buffer);
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
		format_http_request(req, stream);
		stream.flush();
	}
	return stream;
}

std::istream & Request::read_from_stream(std::istream & stream)
{
	stream.peek();
	if (stream.eof())
	{
		return stream;
	}

	HttpRequest req;
	bool success;
	parse_http_request(req, stream, success);

	if (!success)
	{
		stream.setstate(ios::badbit);
		return stream;
	}

	if (req.method == "GET") method = M_GET;
	else if (req.method == "PUT") method = M_PUT;
	else if (req.method == "POST") method = M_POST;
	else if (req.method == "DELETE") method = M_DELETE;
	else if (req.method == "LOCK") method = M_LOCK;
	else if (req.method == "UNLOCK") method = M_UNLOCK;
	else throw bs_exception("Request", "Invalid method.");

	uri = Uri(req.uri);
	
	if (req.headers.contains(http::CONTENT_TYPE))
	{
		string x_type = req.headers.get_value(http::CONTENT_TYPE);
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
			if (req.headers.get_value(http::TRANSFER_ENCODING) == "Chunked")
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
	}

	return stream;
}