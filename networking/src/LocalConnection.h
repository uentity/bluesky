#ifndef __LOCALCONNECTION_H__298DC731_56B4_4504_A0F4_820A447966BE_
#define __LOCALCONNECTION_H__298DC731_56B4_4504_A0F4_820A447966BE_
#pragma once

#include <networking/Connection.h>
#include <networking/Context.h>

namespace blue_sky
{
	namespace networking {

		class NameService;


		// - Обрабатывает сообщения локально (NameService, ResourceManager)
		class LocalConnection : public Connection
		{
			class Impl;
			boost::shared_ptr<Impl> pimpl;			
		public:
			LocalConnection(std::string const& uri, ContextPtr context);

			virtual ~LocalConnection(void);

			virtual void send(
				const Request &,
				Response &);

			virtual void close();
		};
	
	}
}

#endif //__LOCALCONNECTION_H__298DC731_56B4_4504_A0F4_820A447966BE_