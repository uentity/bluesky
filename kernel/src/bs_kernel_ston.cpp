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

//
// C++ Implementation: bs_kernel_ston
//
// Description:
//
//
// Author: Гагарин Александр Владимирович <GagarinAV@ufanipi.ru>, (C) 2008
//
#include "bs_kernel.h"
#include "thread_pool.h"
#include "bs_report.h"
#include "bs_log_scribers.h"
#include "bs_kernel_tools.h"

//#define LOKI_CLASS_LEVEL_THREADING
#include "loki/Singleton.h"

using namespace Loki;
using namespace std;

namespace {
using namespace blue_sky;
/*-----------------------------------------------------------------------------
 *  Specific logging system wrappers
 *-----------------------------------------------------------------------------*/
struct bs_log_wrapper : public bs_log
{
	static bool kernel_alive;

	bs_log_wrapper ()
	{
		if (kernel_alive)
		{
			register_signals ();
		}

		this->add_channel (sp_channel (new bs_channel (OUT_LOG)));
		this->add_channel (sp_channel (new bs_channel (ERR_LOG)));

		char *c_dir = NULL;
		if (!(c_dir = getenv("BS_KERNEL_DIR")))
			c_dir = (char *)".";

		this->get_locked (OUT_LOG, __FILE__, __LINE__).get_channel ()->attach(sp_stream(new log::detail::cout_scriber ("COUT")));
		this->get_locked (OUT_LOG, __FILE__, __LINE__).get_channel ()->attach(sp_stream(new log::detail::file_scriber ("FILE", string(c_dir) + string("/blue_sky.log"), ios::out|ios::app)));
		this->get_locked (ERR_LOG, __FILE__, __LINE__).get_channel ()->attach(sp_stream(new log::detail::cout_scriber ("COUT")));
		this->get_locked (ERR_LOG, __FILE__, __LINE__).get_channel ()->attach(sp_stream(new log::detail::file_scriber ("FILE", string(c_dir) + string("/errors.log"), ios::out|ios::app)));

		this->get_locked (OUT_LOG, __FILE__, __LINE__) << output_time;
		this->get_locked (ERR_LOG, __FILE__, __LINE__) << output_time;
	}

	//static bool &
	//kernel_dead ()
	//{
	//	static bool kernel_dead_ = false;
	//	return kernel_dead_;
	//}
	bs_log & 
	get_log () 
	{
		return *this;
	}

	void
	register_signals ()
	{
		this->add_signal (BS_SIGNAL_RANGE (bs_log));
	}
};

bool bs_log_wrapper::kernel_alive = false;

struct thread_log_wrapper : public thread_log
{
	thread_log_wrapper ()
	{
	}

	thread_log &
	get_log ()
	{
		return *this;
	}

	void
	register_signals ()
	{
	}
	//static bool &
	//kernel_dead ()
	//{
	//	static bool kernel_dead_ = false;
	//	return kernel_dead_;
	//}
};

typedef SingletonHolder < bs_log_wrapper, CreateUsingNew, PhoenixSingleton >      bs_log_holder;
typedef SingletonHolder < thread_log_wrapper, CreateUsingNew, PhoenixSingleton >  thread_log_holder;

} // eof hidden namespace

namespace blue_sky {
namespace bs_private {
//static bool kernel_alive = false;

/// @brief Wrapper allowing to do some initialization on first give_kernel()::Instance() call
/// just after the kernel class is created
struct wrapper_kernel {
	kernel k_;

	kernel& (wrapper_kernel::*ref_fun_)();

	// constructor
	wrapper_kernel()
		: ref_fun_(&wrapper_kernel::initial_kernel_getter)
	{
		bs_log_wrapper::kernel_alive = true;
	}

	// normal getter - just returns kernel reference
	kernel& usual_kernel_getter() {
		return k_;
	}

	// when kernel reference is obtained for the first time
	kernel& initial_kernel_getter() {
		// first switch to usual getter to avoid infinite constructor recursion during load_plugins()
		ref_fun_ = &wrapper_kernel::usual_kernel_getter;
		// initialize kernel
		k_.init();

#ifdef BS_AUTOLOAD_PLUGINS
		// load plugins
		k_.LoadPlugins();
#endif
		// return reference
		return k_;
	}

	kernel& k_ref() {
		return (this->*ref_fun_)();
	}

	~wrapper_kernel() {
		// signal that it is destroyed
		bs_log_wrapper::kernel_alive = false;
	}
};

}	// eof bs_private namespace


/*-----------------------------------------------------------------------------
 *  kernel signleton instantiation
 *-----------------------------------------------------------------------------*/
//! multithreaded kernel singleton - disables
//typedef SingletonHolder< bs_private::wrapper_kernel, CreateUsingNew,
//	DefaultLifetime, ClassLevelLockable > kernel_holder;

// kernel itself is fully multithreaded so we can use simple singleton
// typedef SingletonHolder< bs_private::wrapper_kernel > kernel_holder;

//kernel singletone - master, fabs die after kernel dies
typedef SingletonHolder< bs_private::wrapper_kernel, CreateUsingNew,
	FollowIntoDeath::With< DefaultLifetime >::AsMasterLifetime > kernel_holder;

template< >
BS_API kernel& singleton< kernel >::Instance()
{
	//cout << "give_kernel.Instance() entered" << endl;
	return kernel_holder::Instance().k_ref();
}

//! thread pool singleton
typedef SingletonHolder< blue_sky::worker_thread_pool, CreateUsingNew,
	FollowIntoDeath::AfterMaster< kernel_holder >::IsDestroyed > wtp_holder;

typedef singleton< worker_thread_pool > give_wtp;

template< >
worker_thread_pool& singleton< worker_thread_pool >::Instance() {
	return wtp_holder::Instance();
}

//	void kernel::add_task(const blue_sky::sp_com& task)
//	{
//		give_wtp::Instance().add_command(task);
//	}


/*-----------------------------------------------------------------------------
 *  log singletons instantiation
 *-----------------------------------------------------------------------------*/
typedef singleton <bs_log>      bs_log_singleton;
typedef singleton <thread_log>  thread_log_singleton;

template< >
bs_log& singleton< bs_log >::Instance()
{
	return bs_log_holder::Instance().get_log();
}

template< >
thread_log& singleton< thread_log >::Instance()
{
	return thread_log_holder::Instance().get_log();
}

bs_log& kernel::get_log()
{
	return bs_log_singleton::Instance();
}

thread_log& kernel::get_tlog()
{
	return thread_log_singleton::Instance();
}

}	// namespace blue_sky

