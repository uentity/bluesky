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
 * \threading in blue sky implimentations.
 * \author Andrey Morozov <andrew.morozov@gmail.com icq 213000915>
 * \date 2008-01-30
 */
#ifdef BSPY_EXPORTING_PLUGIN
#include <boost/python.hpp>
#endif

#include "thread_pool.h"
#include <functional>
#include <algorithm>
//DEBUG
#include <iostream>

//number of threads. note it can't be equal to zero!
//when i tested it on intel(R) Core(TM)2 CPU 6600 @ 2.40GHz max number was 1984 ;) over - boost resource error
#define NUM_OF_THREADS 2

// -----------------------------------------------------
// Implementation of class: thread_pool
// -----------------------------------------------------

using namespace std;
using namespace blue_sky;
using namespace boost;


namespace blue_sky {

#if 0
	//Debug
	/*void m_print(std::string name, int num, std::string message)
			{
				static bs_mutex m;
				bs_mutex::scoped_lock lk(m);
				std::cout<<name<<" "<<num<<" : "<<message<<std::endl;
			}*/
#endif

//////////////////////////////////////Self-regulating threads/////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

//thread handler
struct worker_thread
{
	worker_thread(worker_thread_pool &th, int my_number)
		: wtp_(th), n_(my_number)
	{}

	void operator()() {
		while (!wtp_.dying_) {
			//waiting for command in queue
			bs_mutex::scoped_lock wlk(wtp_.worker_mutex_);
			wtp_.worker_condition_.wait(wlk);
			while (true) {
				{
					//locking commands queue
					lsmart_ptr <blue_sky::sp_com_queue> l_sp_commands_queue(wtp_.sp_commands_queue_);
					//m_print("WORKER",n,"I took queue");
					//taking command from queue
					if(l_sp_commands_queue->empty()) break;
					my_com_ = l_sp_commands_queue->front();
					l_sp_commands_queue->pop();
					//m_print("WORKER",n,"I released queue");
					//unlock of command queue
				}

				//chain execution
				while(my_com_)
					my_com_ = my_com_.lock()->execute();
				//m_print("WORKER",n,"finished working");
			}
			//signal that ordinary task is complete
			wtp_.tasks_done_.notify_all();
		}

		cout << "worker " << n_ << " finished" << endl;
	}

private:
	worker_thread_pool &wtp_;
	int n_; //number of this thread
	sp_com my_com_; //command to execute
};

worker_thread_pool::worker_thread_pool()
	: dying_(false)
{
	sp_commands_queue_ = new std::queue<sp_com>;
	{
		//lock of commands queue while threads creation
		lsmart_ptr <blue_sky::sp_com_queue> l_sp_commands_queue(sp_commands_queue_);
		//threads creation
		for(int i = 0; i < NUM_OF_THREADS; ++i) {
			threads_.create_thread(worker_thread(*this,i));
		}
	}
}

worker_thread_pool::~worker_thread_pool() {
	dying_ = true;
	worker_condition_.notify_all();
	cout << "~wtp: notyfy_all signal sent" << endl;
	threads_.join_all();
	cout << "~wtp: all workers stopped" << endl;
}

void worker_thread_pool::add_command(const sp_com& command)
{
	//new command to queue
	sp_commands_queue_.lock()->push(command);
	//informing threads about it
	worker_condition_.notify_one();
}

bool worker_thread_pool::is_queue_empty() const {
	return sp_commands_queue_->empty();
}

//helper structs to bind single functional argument
template< class Op >
class binder_single {
protected:
	Op op_;
	typename Op::argument_type value_;

public:
	//typedef typename Op::result_type result_type;

	binder_single(const Op& operation, const typename Op::argument_type& arg)
		: op_(operation), value_(arg)
	{}

	typename Op::result_type operator()() const {
		return op_(value_);
	}
};

template< class Op, class arg_t >
inline binder_single< Op > bind_single(const Op& op, const arg_t& arg) {
	typedef typename Op::argument_type argument_type;
	return binder_single< Op >(op, argument_type(arg));
}

void worker_thread_pool::wait_queue_empty() const {
	bs_mutex::scoped_lock lk(all_done_);
	//block current thread until all work is done
	tasks_done_.wait(lk, bind_single(mem_fun(&worker_thread_pool::is_queue_empty), this));
}

#if 0
//////////////////////////////////////Controllable threads////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

namespace bs_private {
		//Not Null current command predicate
		class Pred
		{
		public:
			Pred(sp_com &c): c_(c) {}
			bool operator()()
			{
				return (c_!=NULL);
			}
			sp_com &c_;
		};

		//Null current command predicate
		class Pred1
		{
		public:
			Pred1(sp_com &c): c_(c) {}
			bool operator()()
			{
				return (c_==NULL);
			}
			sp_com &c_;
		};
	}


struct controllable_thread
{
	controllable_thread(controllable_thread_pool &th, int my_number) :
		wtp_(th)
		{n=my_number;}
	void operator()()
	{
		bs_private::Pred pr(wtp_.current_command);
	while (true)
	{
		//m_print("WORKER",n,"I'm waiting for masters order");
		wtp_.condition.wait(mutex::scoped_lock(wtp_.mutex,true),pr);
		{
			if(!wtp_.current_command) continue;
			my_com=wtp_.current_command;
			wtp_.current_command=NULL;
			//m_print("WORKER",n,"I took command and free curr_command. I'm running");
			wtp_.loaded_condition.notify_one();
		}
		my_com.lock()->execute(0);
		//m_print("WORKER",n,"I finished");
	}
   }
	controllable_thread_pool &wtp_;
	int n;
	sp_com my_com;
};

///struct for master_thread
struct master_thread
{
	master_thread(controllable_thread_pool &th) :
		wtp_(th)
	{ }
	void operator()()
	{
	//debug counter
	int n=0;
	bs_private::Pred1 pr(wtp_.current_command);
   	while (true)
	{
		//m_print("MASTER",0,"I'm waiting for add command");
		wtp_.master_condition.wait(mutex::scoped_lock(wtp_.master_mutex,true));
		while(!(wtp_.sp_commands_queue->empty()))
		{
			//m_print("MASTER",0,"I'm taking command from queue");
			{
				lsmart_ptr <blue_sky::sp_com_queue> l_sp_commands_queue(wtp_.sp_commands_queue);
				wtp_.current_command=l_sp_commands_queue->front();
				l_sp_commands_queue->pop();
			}
			//m_print("MASTER",n,"I'm waiting while worker getting command");
				wtp_.condition.notify_one();
				wtp_.loaded_condition.wait(mutex::scoped_lock(wtp_.load_mutex,true),pr);
		}
	}
	}
	controllable_thread_pool &wtp_;
};

void controllable_thread_pool::add_command(sp_com command)
{
	sp_commands_queue.lock()->push(command);
	master_condition.notify_one();
}


controllable_thread_pool::controllable_thread_pool()
{
	sp_commands_queue = new std::queue<sp_com>;
	{
		lsmart_ptr <blue_sky::sp_com_queue> l_sp_commands_queue(sp_commands_queue);

		for(int i=0;i<NUM_OF_THREADS;i++)
		{
			controllable_thread controllable_thread_(*this,i);
			threads.push_back(new thread(controllable_thread_));
		}

		master_thread master_thread_(*this);
		sp_master_thread = new thread(master_thread_);
	}
}
#endif

}	//namespace blue_sky

