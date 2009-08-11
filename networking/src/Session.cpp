#include "pch.h"

#include <networking/Session.h>
#include <networking/Path.h>

using namespace std;
using namespace boost;
using namespace blue_sky;
using namespace blue_sky::networking;

class DoubleLock
{
	mutex::scoped_lock lock1;
	mutex::scoped_lock lock2;
public:
	DoubleLock(mutex * muty1, mutex * muty2)
		: lock1(*muty1), lock2(*muty2)
	{

	}
};

typedef boost::shared_ptr<mutex::scoped_lock> LockPtr;
typedef std::map<std::string, LockPtr> LockMap;

typedef shared_ptr<iostream> StreamPtr;



class Session::Impl
{
public:
	mutex l_lockmap;
	LockMap fLocks;
	list<Path> locked_paths;
	condition path_unlocked;
};

Session::Session()
	: pimpl(new Impl())
{
	
}


void Session::lock(const Path & p, mutex * muty)
{
	
	/**/
	LockPtr lp(new mutex::scoped_lock(*muty));
	//std::cout << "Session::locked(" << p.str() << ", " << (unsigned long)muty << ");" << std::endl;
	mutex::scoped_lock lock(pimpl->l_lockmap);
	pimpl->fLocks.insert(make_pair(p.str(), lp));

	/*/
	mutex::scoped_lock lock(pimpl->l_lockmap);
	while (find(pimpl->locked_paths.begin(), pimpl->locked_paths.end(), p) != pimpl->locked_paths.end() )
	{
	pimpl->path_unlocked.wait(lock);
	}
	pimpl->locked_paths.push_back(p);
	/**/
}

bool Session::is_locked(const Path & p)
{
	mutex::scoped_lock lock(pimpl->l_lockmap);
	LockMap::iterator it = pimpl->fLocks.find(p.str());
	return it != pimpl->fLocks.end();
}

void Session::unlock(const Path & p)
{
	//std::cout << "Session::unlock(" << p.str() << ");" << std::endl;
	/**/
	mutex::scoped_lock lock(pimpl->l_lockmap);
	LockMap::iterator it = pimpl->fLocks.find(p.str());
	if (it != pimpl->fLocks.end())
	{
		pimpl->fLocks.erase(it);
	}
	/*/
	mutex::scoped_lock lock(pimpl->l_lockmap);
	pimpl->locked_paths.remove(p);	
	/**/
}



Session::~Session()
{}

class NetworkSession::Impl
{
public:
	StreamPtr fStream;
	bool fIsNull;
};

NetworkSession::NetworkSession()
: pimpl(new Impl())
{
	pimpl->fIsNull = true;
}

NetworkSession::NetworkSession(shared_ptr<iostream> stream)
	: pimpl(new Impl())
{
	pimpl->fIsNull = false;
	pimpl->fStream = stream;	
}

iostream & NetworkSession::stream()
{
	return *(pimpl->fStream);
}

bool NetworkSession::is_null()
{
	return pimpl->fIsNull;
}

NetworkSession::~NetworkSession()
{
}