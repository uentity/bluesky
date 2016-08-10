/// @file
/// @author uentity
/// @date 10.08.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel.h>

//#define LOKI_CLASS_LEVEL_THREADING
#include <loki/Singleton.h>

using namespace Loki;
using namespace std;

namespace blue_sky { namespace detail {
static bool kernel_alive = false;

/// @brief Wrapper allowing to do some initialization on first give_kernel()::Instance() call
/// just after the kernel class is created
struct wrapper_kernel {
	kernel k_;

	kernel& (wrapper_kernel::*ref_fun_)();

	static void kernel_cleanup();

	// constructor
	wrapper_kernel()
		: ref_fun_(&wrapper_kernel::initial_kernel_getter)
	{
		kernel_alive = true;
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
#ifdef BSPY_EXPORTING
		// if we build with Python support
		// register function that cleans up kernel
		// when Python interpreter exits
		//Py_AtExit(&kernel_cleanup);
#endif
		// return reference
		return k_;
	}

	kernel& k_ref() {
		return (this->*ref_fun_)();
	}

	~wrapper_kernel() {
		// signal that it is destroyed
		kernel_alive = false;
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
typedef SingletonHolder< detail::wrapper_kernel, CreateUsingNew,
	FollowIntoDeath::With< DefaultLifetime >::AsMasterLifetime > kernel_holder;

template< >
BS_API kernel& singleton< kernel >::Instance() {
	//cout << "give_kernel.Instance() entered" << endl;
	return kernel_holder::Instance().k_ref();
}

void detail::wrapper_kernel::kernel_cleanup() {
	BS_KERNEL.cleanup();
}

}	// namespace blue_sky

