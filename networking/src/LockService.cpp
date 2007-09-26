#include "pch.h"
#include "LockService.h"

using namespace std;
using namespace boost;
using namespace blue_sky;
using namespace blue_sky::networking;

struct LockStruct
{
	int token;
	boost::shared_ptr<boost::condition> unlocked;
	LockStruct()
		: unlocked(new condition())
	{}
};

class LockService::Impl
{
public:
	typedef std::map<std::string, LockStruct> LockMap;
	LockMap fMap;
	boost::mutex mutex;
	boost::mutex mutex2;
	boost::condition unlocked;
};



LockService::LockService()
: pimpl(new Impl())
{

}

LockService::~LockService()
{

}

int LockService::lock(const std::string &path)
{
	static int token_gen = 1;
	mutex::scoped_lock lock(pimpl->mutex);
	for (;;)
	{
		Impl::LockMap::iterator it = pimpl->fMap.find(path);
		if (it == pimpl->fMap.end())
		{
			int token = token_gen++;
			LockStruct ls;
			pimpl->fMap.insert(make_pair(path, ls));
			return token;
		}
		else
		{
			pimpl->unlocked.wait(lock);
		}
	}	
}

void LockService::unlock(const std::string &path, int lock_token)
{
	mutex::scoped_lock lock(pimpl->mutex);
	Impl::LockMap::iterator it = pimpl->fMap.find(path);
	if (it != pimpl->fMap.end())
	{
		shared_ptr<condition> spcon = it->second.unlocked;
		pimpl->fMap.erase(it);
		
		pimpl->unlocked.notify_all();
	}	
}