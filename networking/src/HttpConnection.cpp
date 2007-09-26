#include "pch.h"

#ifdef _MSC_VER
#pragma warning(disable:6328 6246)
#endif

#include <boost/asio.hpp>

#ifdef _MSC_VER
#pragma warning(default:6328 6246)
#endif

//NETWORKING
#include <networking/ISerializable.h>
#include "HttpConnection.h"
#include <networking/ServerSettings.h>

//HTTP_LIB
#include <Http.h>
#include <HttpRequest.h>
#include <HttpResponse.h>
//#include <HttpLibraryException.h>
#include <networking/Uri.h>

#include <networking/ClientSettings.h>

#include <http/Request.h>
#include <Response.h>

//#include "Cache.h"


using namespace std;
using namespace boost;

using namespace boost::asio;
using namespace boost::asio::ip;
using namespace blue_sky;
using namespace blue_sky::networking;
using namespace blue_sky::http_library;

class HttpConnection::Impl
{
public:
	bool is_closed_;				
	ContextPtr context;
	boost::shared_ptr<tcp::iostream> network_stream_;	
};

HttpConnection::HttpConnection(
	ContextPtr context, 	
	const std::string &uri)
	: Connection(uri), 
	pimpl(new HttpConnection::Impl())	
{
	pimpl->context = context;
	pimpl->is_closed_ = false;
	Uri u(uri);
	pimpl->network_stream_.reset(new tcp::iostream(u.host(), u.port()));	
}

void HttpConnection::close()
{	
	pimpl->network_stream_->close();
	pimpl->is_closed_ = true;
}

HttpConnection::~HttpConnection()
{
	if (!pimpl->is_closed_)
		close();
}

namespace {
//Returns message "<STATUS CODE> <Reason phrase>". Ex.: "404 Not found."
void format_message(const HttpResponse & resp, string & message)
{
	message = str(format("%1% %2%") % resp.status_code % resp.reason_phrase);	
}
}

void HttpConnection::send(
	const Request  & req, 
	      Response & resp)
{	
//	network_stream_->peek();
	
	if (!pimpl->network_stream_->good())
	{
		throw bs_exception("HttpConnection", "Connection was terminated.");
	}
//	std::cerr << "HttpConnection::send" << endl;
		
//		std::cerr << "Sending request... ";
		
	*(pimpl->network_stream_) << req;
	pimpl->network_stream_->flush();

//		std::cerr << "done." << std::endl;


//		std::cerr << "Reading response...";
	*(pimpl->network_stream_) >> resp;

	if (pimpl->network_stream_->bad())
		throw bs_exception("HttpConnection", "Invalid http response.");	
}