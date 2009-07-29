#ifndef __LOCKSERVICE_H__BF0E76C2_7DD2_455E_B408_12B87FDB06F6__
#define __LOCKSERVICE_H__BF0E76C2_7DD2_455E_B408_12B87FDB06F6__

#include <boost/shared_ptr.hpp>

namespace blue_sky
{
	namespace networking
	{
		class LockService
		{
			class Impl;
			boost::shared_ptr<Impl> pimpl;
		public:
			LockService();
			~LockService();
			int lock(const std::string & path);
			void unlock(const std::string & path, int lock_token);
		};
	}
}

#endif //__LOCKSERVICE_H__BF0E76C2_7DD2_455E_B408_12B87FDB06F6__