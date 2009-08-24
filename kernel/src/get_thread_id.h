/**
 * \file get_thread_id.h
 * \brief returns current thread id
 * \author Sergey Miryanov
 * \date 14.08.2009
 * */
#ifndef BS_GET_THREAD_ID_H_
#define BS_GET_THREAD_ID_H_

#ifdef UNIX
	#include <pthread.h>
#else
//#elif defined(_WIN32) && defined(_MSC_VER)
	#include "windows.h"
#endif

namespace blue_sky {
namespace detail {

#ifndef UNIX
    static unsigned long int
    get_thread_id ()
    {
      return (unsigned long int)GetCurrentThreadId ();
    }
#else
    static unsigned long int
    get_thread_id ()
    {
      return (unsigned long int)pthread_self();
    }
#endif

//	unsigned long int get_thread_id() {
//#ifdef UNIX
//		return (unsigned long int)pthread_self();
//#elif defined(_WIN32) && defined(_MSC_VER)
//		return (unsigned long int)GetCurrentThreadId();
//#endif
//	}

} // namespace detail
} // namespace blue_sky


#endif // #ifndef BS_GET_THREAD_ID_H_

