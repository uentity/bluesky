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

//#define LOKI_CLASS_LEVEL_THREADING
#include "loki/Singleton.h"

using namespace Loki;

namespace blue_sky {

	namespace bs_private {

		struct wrapper_kernel {

			 kernel k_;

//#ifdef BS_AUTOLOAD_PLUGINS
			 kernel& (wrapper_kernel::*ref_fun_)();

			 //constructor
			 wrapper_kernel()
				 : ref_fun_(&wrapper_kernel::initial_kernel_getter)
			 {}

			 //normal getter - just returns kernel reference
			 kernel& usual_kernel_getter() {
				 return k_;
			 }

			 //when kernel reference is obtained for the first time - load plugins
			 kernel& initial_kernel_getter() {
				 //first switch to usual getter to avoid infinite constructor recursion during load_plugins()
				 ref_fun_ = &wrapper_kernel::usual_kernel_getter;
				 //initialize kernel
				 k_.init();

#ifdef BS_AUTOLOAD_PLUGINS
				 //load plugins
				 k_.LoadPlugins();
#endif
				 //return reference
				 return k_;
			 }

			 kernel& k_ref() {
				return (this->*ref_fun_)();
			 }

// 			 kernel& k_ref() {
// 				 return k_;
// 			 }
		};

	}	// namespace bs_private

	//kernel itself is fully multithreaded so we can use simple singleton
	//typedef SingletonHolder< bs_private::wrapper_kernel > kernel_holder;

	//! standard multithreaded kernel singleton
	//typedef SingletonHolder< bs_private::wrapper_kernel, CreateUsingNew,
	//	DefaultLifetime, ClassLevelLockable > kernel_holder;

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

	//----------------------factories--------------------------------------
	//! Loki object factory type
	//typedef Factory< objbase, BS_TYPE_OBJ, LOKI_TYPELIST_1(bs_type_ctor_param) > obj_factory;
	//typedef Factory< objbase, BS_TYPE_OBJ, LOKI_TYPELIST_1(bs_type_ctor_param) > com_factory;

	//cause kernel is a singletone, fabs can be made kernel_impl members
	//now make multi-threaded singletones from fabs
	//typedef SingletonHolder< obj_factory, CreateUsingNew, FollowIntoDeath::AfterMaster< kernel_holder >::IsDestroyed,
	//		   ClassLevelLockable > obj_fab_holder;
	//typedef SingletonHolder< obj_factory_int, CreateUsingNew, FollowIntoDeath::AfterMaster< kernel_holder >::IsDestroyed,
	//		   ClassLevelLockable > obj_fab_holder_int;
	//typedef SingletonHolder< com_factory, CreateUsingNew, FollowIntoDeath::AfterMaster< kernel_holder >::IsDestroyed,
	//		   ClassLevelLockable > com_fab_holder;

}
