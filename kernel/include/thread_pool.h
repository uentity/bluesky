// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

/*!
 * \file bs_threading.cpp
 * \threading in blue sky.
 * \author Andrey Morozov <andrew.morozov@gmail.com icq 213000915>
 * \date 2008-01-30
 */

#ifndef _THREAD_POOL_H
#define _THREAD_POOL_H

#include "bs_common.h"
#include "bs_command.h"
#include "boost/thread/thread.hpp"
#include "boost/thread/condition.hpp"
#include <queue>

namespace blue_sky
{
	typedef smart_ptr < std::queue< sp_com > > sp_com_queue;
	typedef smart_ptr < boost::thread > sp_thread;
	typedef std::vector < sp_thread > sp_thread_vector;

	 /*!
		\brief Thread handler structure.
	*/
	struct worker_thread;

	/*!
		\class worker_thread_pool ... realization of self-regulating threads
	*/
	class worker_thread_pool
	{
	friend struct worker_thread;
	private:
		bool dying_;

		/*!
			\brief Vector of threads
		*/
		boost::thread_group threads_;
		/*!
			\brief Mutex for thread handlers
		*/
		blue_sky::bs_mutex worker_mutex_;
		/*!
			\brief the condition of not empty command queue
		*/
		boost::condition worker_condition_;
		/*!
			\brief the queue of commands
		*/
		sp_com_queue sp_commands_queue_;

		//mutex and condition for waiting for all task completiotion
		mutable bs_mutex all_done_;
		mutable boost::condition tasks_done_;

		void join_kill();

	public:
		/*!
			\brief constructor
		*/
		worker_thread_pool();
		~worker_thread_pool();
		/*!
			\brief push new command to queue
		*/
		void add_command(const sp_com& command);

		//function for checking if all work is done
		//returns immediately
		bool is_queue_empty() const;

		//function for waiting until all work is done
		//will block calling thread until queue become empty
		void wait_queue_empty() const;
	};

#if 0
	/*!
		\class controllable_thread_pool ... realization of threads, which are controlled by master_thread
	*/

	struct controllable_thread;
	struct master_thread;

	class controllable_thread_pool
	{
	friend struct controllable_thread;
	friend struct master_thread;
	private:
		sp_thread sp_master_thread;
		sp_thread_vector threads;

		blue_sky::bs_mutex master_mutex;
		blue_sky::bs_mutex mutex;
		blue_sky::bs_mutex load_mutex;

		boost::condition condition;
		boost::condition master_condition;
		boost::condition loaded_condition;

		sp_com_queue sp_commands_queue;
		sp_com current_command;
		public:

		controllable_thread_pool();
		void add_command(sp_com command);
	};

#endif

}	//blue_sky namespace

#endif //_THREAD_POOL_H
