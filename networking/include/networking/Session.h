#ifndef __SESSION_H__355956F6_D50E_4338_8840_E470C619B39D__
#define __SESSION_H__355956F6_D50E_4338_8840_E470C619B39D__

#include <iostream>
#include <boost/smart_ptr.hpp>
#include <boost/thread.hpp>

namespace blue_sky
{
	namespace networking
	{
		class Path;

		class Session
		{
			class Impl;
			boost::shared_ptr<Impl> pimpl;
		public:
			Session();			
			void lock(const Path & p, boost::mutex * muty);
			void unlock(const Path & p);	
			bool is_locked(const Path & p);
			virtual ~Session();
		};

		class NetworkSession : public Session
		{
			class Impl;
			boost::shared_ptr<Impl> pimpl;
		public:
			NetworkSession();
			NetworkSession(boost::shared_ptr<std::iostream> stream);
			std::iostream & stream();
			virtual bool is_null();	
			virtual ~NetworkSession();
		};
	}
		
}

#endif //__SESSION_H__355956F6_D50E_4338_8840_E470C619B39D__