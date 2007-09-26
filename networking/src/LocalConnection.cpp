#include "pch.h"
#include <networking/Uri.h>
#include <networking/NameService.h>
#include <networking/ResourceManager.h>
//#include <networking/SessionContext.h>

#include <networking/Context.h>
#include <networking/Session.h>

#include "LocalConnection.h"

#include <Request.h>
#include <Response.h>
#include <Http.h>

using namespace std;
using namespace blue_sky;
using namespace blue_sky::networking;
using namespace blue_sky::http_library;

class LocalConnection::Impl
{
public:
	ContextPtr context;
	NameService * name_service;
	std::string path;
	Session session;
};

LocalConnection::LocalConnection(string const& uri, ContextPtr context)
: Connection(uri), pimpl(new Impl())
{
	pimpl->context = context;
	Uri u(uri);
	pimpl->path = u.path();	
}

LocalConnection::~LocalConnection(void)
{
}

void LocalConnection::send(const Request & req, Response & resp)
{	
	try
	{	
		ResourceManager * resource_manager;
		string suffix;	
		pimpl->context->name_service->lookup(req.uri.path(),
			&resource_manager,
			suffix);			
			
		resource_manager->process_request(
			pimpl->session, 
			suffix, 
			req, 
			resp);		
	} catch (std::exception & ex) {		
		resp.Status_code = http::STATUS_500_Internal_Server_Error;
		resp.Reason_Phrase = string("Internal Error: ") + ex.what();		
		resp.Body = 0;
	}
}

void LocalConnection::close()
{
}
