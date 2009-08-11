#ifndef __TREERESOURCEMANAGER_H__4D53D78B_8EEB_4C13_93EA_965EA1A81515_
#define __TREERESOURCEMANAGER_H__4D53D78B_8EEB_4C13_93EA_965EA1A81515_

#include <string>

#include <networking/lib.h>
#include <networking/ResourceManager.h>
#include <networking/Context.h>
#include <networking/Session.h>

namespace blue_sky
{
	namespace networking
	{
		class  BSN_API TreeResourceManager : public ResourceManager
		{
			class Impl;
			boost::shared_ptr<Impl> pimpl;

			TreeResourceManager(const TreeResourceManager &);
			TreeResourceManager & operator=(const TreeResourceManager &);
		public:
			TreeResourceManager(
				ContextPtr context,
				const std::string & path_prefix
			);
		
			virtual void process_request(
				          Session & session,
				       const std::string & path,
				const Request & req,
				     Response & resp
			);

			
		};
	}
}

#endif //__TREERESOURCEMANAGER_H__4D53D78B_8EEB_4C13_93EA_965EA1A81515_
