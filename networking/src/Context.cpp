#include "pch.h"
#include <networking/Context.h>


//using namespace std;
using namespace boost;
using namespace blue_sky;
using namespace blue_sky::networking;

Context::Context()
:
	port(80),
	thread_count(10),
	buffer_size(0x400),
	queue_size(20)
{
}

shared_ptr<Context> Context::create()

				
{
	shared_ptr<Context> result(new Context());
	result->ns_.reset(new NameService());
	result->name_service = &*(result->ns_);	
	result->kernel = &give_kernel::Instance();
	//result->server_settings.reset(new ServerSettings);
	result->client_settings.reset(new ClientSettings);
	return result;
}

