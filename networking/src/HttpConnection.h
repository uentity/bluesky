#ifndef __HTTPCONNECTION_H__D3945292_6E9E_460D_92D7_068B8AF6F4FD_
#define __HTTPCONNECTION_H__D3945292_6E9E_460D_92D7_068B8AF6F4FD_

//crt
#include <memory>

//boost
#include <boost/smart_ptr.hpp>

//bs
#include <bs_kernel.h>

//bsn
#include <networking/Connection.h>

#include <networking/Context.h>

namespace blue_sky
{
	namespace networking
	{	
		class ClientSettings;
		// - Пересылает сообщения по протоколу Http (kernel, ClientSettings, Request, Response)
		class HttpConnection : public Connection
		{
			
			class Impl;
			boost::shared_ptr<Impl> pimpl;
		public:
			HttpConnection(
				ContextPtr context, 				
				const std::string & uri
			);

			virtual ~HttpConnection();

		public:
			//Connection overrides

			virtual void send(
				const Request &,
				Response &
			);
			virtual void close();
		};		
	}
}

#endif //__HTTPCONNECTION_H__D3945292_6E9E_460D_92D7_068B8AF6F4FD_