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
	//assignment operator - does nothing
	//its meaningless to assign 2 refcounters
	bs_refcounter& operator=(const bs_refcounter& /*src*/) { return *this; }

public:
	/*!
	\brief Add reference.
	*/
	virtual void add_ref() const {
		++refcnt_;
	}

	/*!
	\brief Delete reference.
	*/
	virtual void del_ref() const {
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
}	//namespace blue_sky

#endif
