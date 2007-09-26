#ifndef __NAMESERVICE_H__EAF609BA_6A9D_4E4A_AC02_BBC81F5E116C_
#define __NAMESERVICE_H__EAF609BA_6A9D_4E4A_AC02_BBC81F5E116C_

#include <string>
#include <map>
#include <iosfwd>

#include <boost/thread.hpp>

namespace blue_sky
{
	namespace networking {
		class ResourceManager;

		struct LookupResult {
			ResourceManager * resource_manager;
			std::string suffix;
		};

		// - Регистрирует менеджеры ресурсов (ResourceManager)
		// - Находит менеджера ресурса по заданному пути		
		class NameService : boost::noncopyable
		{	
			mutable boost::mutex map_mutex_;
			std::map<std::string, ResourceManager *> map_;
			friend class ResourceManager;
			//NameService(const NameService&);
		private:

			//registers resource manager
			void add(ResourceManager * manager);

			//unregisters resource manager by prefix
			void remove(std::string const & path_prefix);

			//unregisters resource manager
			void remove(ResourceManager * manager);
		public:			
			//returns resource manager responsible for path and <suffix> = <path> - <resource manager prefix>
			bool lookup(std::string const& path, ResourceManager ** rm, std::string & suffix) const;

			//debug dump
			void dump(std::ostream & stream) const;
		};

		//debug dump
		std::ostream & operator<<(std::ostream & stream, NameService const& service);
	}

	
}

#endif //__NAMESERVICE_H__EAF609BA_6A9D_4E4A_AC02_BBC81F5E116C_