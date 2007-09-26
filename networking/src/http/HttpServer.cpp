#include "pch.h"

#include <HttpParsing.h>
#include <HttpFormatting.h>

#include <networking/HttpServer.h>
#include "TaskQueue.h"
#include <networking/Context.h>
#include <networking/ServerSettings.h>
#include <networking/Session.h>
#include <networking/ResourceManager.h>
#include <Request.h>
#include <Response.h>
#include <BadRequestException.h>

using namespace std;
using namespace boost;
using namespace boost::asio;
using namespace boost::asio::ip;
using namespace blue_sky;
using namespace blue_sky::http_library;
using namespace blue_sky::networking;

struct WorkerThread
{		
	TaskQueue * fQueue;
	int id;
	bool stopping;
	networking::ContextPtr fContext;
public:

	WorkerThread(
		networking::ContextPtr context,		
		TaskQueue * queue) 
		: 
			fContext(context),			
			fQueue(queue),
			stopping(false)
	{
		static int last_id = 0;
		id = ++last_id;		
	}

	void handle_request(NetworkSession & session)
	{	
		Request request(fContext);
		Response response(fContext);

		Uri uri;
		ResourceManager * resource_manager;
		string suffix;	

		session.stream() >> request;
		if (session.stream().bad())
			throw BadRequestException();

		uri = request.uri;						

		fContext->name_service->lookup(				
			request.uri.path(), 
			&resource_manager, 
			suffix);			

		resource_manager->process_request(
			session,
			suffix, 
			request, 
			response);						

		session.stream() << response;			
	}

	void operator()()
	{		
		for (;;) {
			try
			{
				TaskQueue::QueueItem item = fQueue->get();								
				if (item.is_null() == false)
				{
					item.stream().peek();
					if (item.stream().good())
					{
						handle_request(item);
						fQueue->put(item);
					}
					else
						// Connection closed.
						continue;
				}
				else
				{		
					//Server shutdown
					return;
				}				
			}			
			catch (exception & ex)
			{
				std::cerr << "HttpServer: " << ex.what() << std::endl;
			}
		}
	}	
};

class AcceptingThread
{
	tcp::acceptor & fAcceptor;
	TaskQueue & fQueue;
public:
	AcceptingThread(tcp::acceptor & acceptor, TaskQueue & queue)
		: fAcceptor(acceptor),
		  fQueue(queue)
	{}

	void operator()()
	{
		for (;;)
		{
			shared_ptr<std::iostream> stream(new tcp::iostream());
			tcp::iostream * tcpstream = (tcp::iostream *) stream.get();
			fAcceptor.accept(*(tcpstream->rdbuf()));
			TaskQueue::QueueItem item(stream);		
			fQueue.put(item);
		}
	}

	void stop()
	{
		fQueue.stop();
	}
};

class HttpServer::Impl
{	
	networking::ContextPtr fContext;
	io_service service;
	TaskQueue queue;
	boost::thread_group threadgroup;
	std::vector<shared_ptr<WorkerThread> > threads;
	tcp::acceptor acceptor;
	
	AcceptingThread fAcceptingThreadProc;
	boost::thread fAcceptingThread;
private:
	Impl(const Impl&);
public:			
	Impl(
		networking::ContextPtr context)
		: queue(context->queue_size),
		fContext(context),
		acceptor(service, tcp::endpoint(tcp::v4(), context->port)),
		fAcceptingThreadProc(acceptor, queue),
		fAcceptingThread(fAcceptingThreadProc)
	{
		
	}

	int get_thread_count()
	{
		return fContext->thread_count;
	}

	virtual void start()
	{		
		for (int i = 0, end_i = get_thread_count(); i < end_i; ++i)
		{
			shared_ptr<WorkerThread> thr(
				new WorkerThread(fContext, &queue)
			);
			threads.push_back(thr);
			threadgroup.create_thread(*thr);			
		}		
	}

	virtual void begin_stop()
	{
		fAcceptingThreadProc.stop();		
	}

	virtual void end_stop()
	{
		threadgroup.join_all();
	}

	virtual void stop()
	{
		begin_stop();
		end_stop();
	}	
};

HttpServer::HttpServer(ContextPtr context)
: pimpl(new HttpServer::Impl(context))
{
}

HttpServer::~HttpServer()
{}

void HttpServer::start()
{	
	pimpl->start();
}

void HttpServer::stop()
{
	pimpl->stop();
}

void HttpServer::begin_stop()
{
	pimpl->begin_stop();
}

void HttpServer::end_stop()
{
	pimpl->end_stop();
}





