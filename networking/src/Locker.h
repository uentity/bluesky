#ifndef __LOCKER_H__638F8088_F394_4865_A2FD_F3F55D617827__
#define __LOCKER_H__638F8088_F394_4865_A2FD_F3F55D617827__

namespace blue_sky
{
	namespace networking
	{
		class Locker
		{
		public:
			void start();
			void stop();
			void lock(boost::mutex * muty);
			void unlock(boost::mutex * muty);
		}
	}
}

#endif //__LOCKER_H__638F8088_F394_4865_A2FD_F3F55D617827__