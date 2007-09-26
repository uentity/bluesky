#ifndef __TASKQUEUE_H__991AA259_7FD0_4B7E_B721_B0CA3F02F4D4_
#define __TASKQUEUE_H__991AA259_7FD0_4B7E_B721_B0CA3F02F4D4_

#include <memory>
#include <vector>
#include <boost/smart_ptr.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/asio.hpp>
#include <boost/noncopyable.hpp>

namespace blue_sky
{
	namespace networking
	{
		class NetworkSession;
		

		class TaskQueue : public boost::noncopyable
		{			
			TaskQueue(const TaskQueue&);
		public:
			typedef NetworkSession QueueItem;
		private:
			QueueItem * queue_;
			int first_allocated_;
			int first_free_;
			int allocated_;
			int size_;
			boost::mutex mutex_;
			boost::condition buffer_not_full_;
			boost::condition buffer_not_empty_;

			bool stopping_;
		public:
			TaskQueue(int size);
			~TaskQueue();

			// returns false if stop() is called.
			QueueItem get(); 
			void put(QueueItem item);		
			void stop();
		};
	}
}

#endif //__TASKQUEUE_H__991AA259_7FD0_4B7E_B721_B0CA3F02F4D4_