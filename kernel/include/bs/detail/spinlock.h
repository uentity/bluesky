/// @file
/// @author uentity
/// @date 10.01.2020
/// @brief User-space spinlog impl (yes, ugly idea, but intended to be used only on Windows)
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

// [NOTE] !!! USE ONLY ON WINDOWS !!!
// Linux (and other POSIX) tends to have very good futexes as backend for `std::mutex`
//
// References:
// 1. https://probablydance.com/2019/12/30/measuring-mutexes-spinlocks-and-how-bad-the-linux-scheduler-really-is/
// 2. https://www.realworldtech.com/forum/?threadid=189711&curpostid=189723
// and discussion there

#include <atomic>
#include <boost/smart_ptr/detail/yield_k.hpp>

namespace blue_sky::detail {

struct spinlock {
	void lock() {
		for(unsigned spin_count = 0; !try_lock(); ++spin_count)
			boost::detail::yield(spin_count);
	}

	bool try_lock() {
		return !locked.load(std::memory_order_relaxed) && !locked.exchange(true, std::memory_order_acquire);
	}

	void unlock() {
		locked.store(false, std::memory_order_release);
	}

private:
	std::atomic<bool> locked{false};
};

} // eof blue_sky::detail
