/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Reference counter class definition for BlueSky types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _BS_REFCOUNTER_H
#define _BS_REFCOUNTER_H

#include "setup_common_api.h"
//replaced simple counter with boost's thread-safe atomic count
#include "boost/detail/atomic_count.hpp"

//mutex includes
//#ifdef UNIX
//define this to enable use of recursive mutex concept
#define BS_FORCE_RECURSIVE_MUTEX
//#endif

#ifdef BS_FORCE_RECURSIVE_MUTEX
#include "boost/thread/recursive_mutex.hpp"
#else
#include "boost/thread/mutex.hpp"
#endif

namespace blue_sky {

//! mutex type used in blue-sky
#ifdef BS_FORCE_RECURSIVE_MUTEX
typedef boost::recursive_mutex bs_mutex;
#undef BS_FORCE_RECURSIVE_MUTEX
#else
typedef boost::mutex bs_mutex;
#endif

class BS_API bs_refcounter {
public:
	//default ctor
	bs_refcounter() : refcnt_(0) {}
	//copy ctor
	bs_refcounter(const bs_refcounter& /*src*/) : refcnt_(0) {}

	// virtual dtor
	virtual ~bs_refcounter() {}

	//assignment operator - does nothing
	//its meaningless to assign 2 refcounters
	bs_refcounter& operator=(const bs_refcounter& /*src*/) { return *this; }

public:
	/*!
	\brief Add reference.
	*/
	void add_ref() const {
		++refcnt_;
	}

	/*!
	\brief Delete reference.
	*/
	void del_ref() const {
		if(--refcnt_ == 0)
			dispose();
	}

	/*!
	\brief Returns references count.
	*/
	long refs() const { return refcnt_; }

	/*!
	\brief Self-destruction method - default is 'delete this', assumes creating with 'new'
	*/
	virtual void dispose() const = 0;
	//{ delete this; }

	/*!
	\brief Mutex accessor.
	\return Reference to mutex
	*/
	bs_mutex& mutex() const { return mut_; }

protected:
	// ctor with given counter
	bs_refcounter(long refcnt) : refcnt_(refcnt) {}
	//mutex for non-const members access
	mutable bs_mutex mut_;

private:
	//reference counter
	mutable boost::detail::atomic_count refcnt_;
};

/*-----------------------------------------------------------------
 * single-threaded reference counter without mutex
 *----------------------------------------------------------------*/
class BS_API bs_refcounter_st {
public:
	// default ctor
	bs_refcounter_st() : refcnt_(0) {}
	// copy ctor
	bs_refcounter_st(const bs_refcounter_st&) : refcnt_(0) {}

	// virtual dtor
	virtual ~bs_refcounter_st() {}

	//assignment operator - does nothing
	//its meaningless to assign 2 refcounters
	bs_refcounter_st& operator=(const bs_refcounter_st& /*src*/) { return *this; }

	/*!
	\brief Add reference.
	*/
	void add_ref() const {
		++refcnt_;
	}

	/*!
	\brief Delete reference.
	*/
	void del_ref() const {
		if(--refcnt_ == 0)
			dispose();
	}

	/*!
	\brief Returns references count.
	*/
	long refs() const { return refcnt_; }

	/*!
	\brief Self-destruction method - default is 'delete this', assumes creating with 'new'
	*/
	virtual void dispose() const = 0;

protected:
	// ctor with given counter
	bs_refcounter_st(long refcnt) : refcnt_(refcnt) {}

private:
	// counter itself
	mutable long refcnt_;
};

}	//namespace blue_sky

#endif
