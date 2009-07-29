#ifndef __HTTP_SERVER_H___332C49C4_5EE7_4F05_823D_6A0D0C7EE242
#define __HTTP_SERVER_H___332C49C4_5EE7_4F05_823D_6A0D0C7EE242

#include <memory>
#include <boost/smart_ptr.hpp>

#include <networking/lib.h>
#include <networking/Context.h>
#include <networking/ServerSettings.h>

namespace blue_sky
{
	namespace networking
	{
		class HttpRequestHandler;	
				
		class BSN_API HttpServer
		{
			class Impl;
			boost::shared_ptr<Impl> pimpl;
		public:		
			HttpServer(blue_sky::networking::ContextPtr context);
			void start();
			void begin_stop();
			void end_stop();
			void stop();
			~HttpServer();
		};	
	}
}

#endif //__HTTP_SERVER_H___332C49C4_5EE7_4F05_823D_6A0D0C7EE242