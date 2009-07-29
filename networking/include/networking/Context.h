#ifndef __CONTEXT_H__E92983CF_3707_4469_B204_E46531B2638E_
#define __CONTEXT_H__E92983CF_3707_4469_B204_E46531B2638E_

#include <boost/smart_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <bs_kernel.h>
#include <networking/lib.h>
#include <networking/NameService.h>
#include <networking/ServerSettings.h>
#include <networking/ClientSettings.h>

namespace blue_sky
{
	
	namespace networking
	{
		class NameService;		
		class ServerSettings;
		class ClientSettings;
		
		class Context;
			

		typedef boost::shared_ptr<Context> ContextPtr;
		


		//Хранит всебе общие настройки и классы
		class  BSN_API Context : public boost::enable_shared_from_this<Context>
		{
		private:
			boost::shared_ptr<NameService> ns_;			
			
		public:			
			Context();
			NameService * name_service;			
			blue_sky::kernel * kernel;
			//ServerSettingsPtr server_settings;			
			ClientSettingsPtr client_settings;
			static ContextPtr create();		

			int port;
			int thread_count;
			int buffer_size;
			int queue_size;
		};

	}
}

#endif //__CONTEXT_H__E92983CF_3707_4469_B204_E46531B2638E_