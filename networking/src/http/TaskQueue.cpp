#include "pch.h"

#include <networking/Session.h>

#include "TaskQueue.h"

using namespace std;
using namespace boost;
using namespace boost::asio;
using namespace boost::asio::ip;

using namespace blue_sky;
using namespace blue_sky::networking;

TaskQueue::TaskQueue(int size)
: 
	first_allocated_(0), 
	first_free_(0), 
	allocated_(0), 
	size_(size),
	stopping_(false)
{
	queue_ = new QueueItem[size];	
}

TaskQueue::~TaskQueue()
{
	delete [] queue_;
}

void TaskQueue::put(QueueItem item)
{
	boost::mutex::scoped_lock lock(mutex_);
	if (stopping_)
		return;
	while (allocated_ == size_)
		buffer_not_full_.wait(lock);
	queue_[first_free_] = item;
	first_free_ = (first_free_ + 1) % size_;
	++allocated_;
	buffer_not_empty_.notify_one();
}

TaskQueue::QueueItem TaskQueue::get()
{
	boost::mutex::scoped_lock lock(mutex_);
	while (allocated_ == 0 && !stopping_)
		buffer_not_empty_.wait(lock);
	if (stopping_)
	{		
		return QueueItem();
	}
	QueueItem result = queue_[first_allocated_];
	queue_[first_allocated_] = QueueItem();
	first_allocated_ = (first_allocated_ + 1) % size_;
	--allocated_;
	buffer_not_full_.notify_one();
	return result;
}

void TaskQueue::stop()
{
	//TODO: Синхронизация
	stopping_ = true;
	buffer_not_empty_.notify_all();
	buffer_not_full_.notify_all();
}
