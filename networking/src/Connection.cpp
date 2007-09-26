#include "pch.h"

#include <networking/Connection.h>
#include "LocalConnection.h"
#include "HttpConnection.h"
#include "networking/Uri.h"
#include <networking/Context.h>

using namespace std;
using namespace blue_sky;
using namespace blue_sky::networking;
using namespace boost;

Connection::Connection(string const& uri)
: uri_(uri)
{
	
}

string const& Connection::uri()
{
	return uri_;
}

Connection::~Connection()
{
	
}

ConnectionPtr Connection::connect(ContextPtr context, std::string uri)
{
	Uri u(uri);
	if (u.protocol() == "http")
		return shared_ptr<Connection>(new HttpConnection(context, uri));
	else if (u.protocol() == "local")
		return shared_ptr<Connection>(new LocalConnection(uri, context));
	else
	{
		string message = "Invalid protocol: ";
		message += u.protocol();
		throw bs_exception("Connection", message.c_str());
	}
}







