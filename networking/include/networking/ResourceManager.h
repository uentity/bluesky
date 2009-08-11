#ifndef RESOURCE_MANAGER_H_E4D089DC_E98F_47DD_9F0E_978B21347093
#define RESOURCE_MANAGER_H_E4D089DC_E98F_47DD_9F0E_978B21347093

#include <string>
#include <boost/shared_ptr.hpp>
#include <networking/lib.h>

namespace blue_sky
{
namespace networking
{	
	class NameService;
	class SessionContext;
	class Request;
	class Response;
	class Session;

	//Базовый класс для менеджеров ресурса
	// - регистрируется и дерегестриуется в службе имен (NameService)
	class  BSN_API ResourceManager
	{		
		std::string path_prefix_;
		NameService * name_service_;
	public:		
		ResourceManager(NameService * store, std::string const & path_prefix);
		virtual ~ResourceManager();

		const std::string & path_prefix();		

		virtual void process_request(
			Session & session,
			std::string const& path,
			const Request & req,
			Response & resp) = 0;
		
	};	
	
}
}

#endif