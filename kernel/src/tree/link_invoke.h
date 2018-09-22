/// @file
/// @author uentity
/// @date 21.09.2018
/// @brief Atomically invoke link methods and correctly manage status
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/common.h>
#include <bs/error.h>
#include <bs/tree/link.h>
#include <bs/tree/errors.h>

#include <atomic>
#include <mutex>
#include <boost/smart_ptr/detail/yield_k.hpp>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree) NAMESPACE_BEGIN(detail)

// microsleep until flag is released by someone else
inline auto raise_atomic_flag(std::atomic_flag& flag) -> void {
	for(unsigned k = 0; flag.test_and_set(std::memory_order_acquire); ++k)
		boost::detail::yield(k);
}

inline auto drop_atomic_flag(std::atomic_flag& flag) {
	flag.clear(std::memory_order_release);
}

// RAII raise-drop atomic flag that keeps track of raised state
struct scope_atomic_flag {
	explicit scope_atomic_flag(std::atomic_flag& flag, bool raise_now = true)
		: f_(flag)
	{
		if(raise_now) raise();
	}

	~scope_atomic_flag() {
		drop();
	}

	inline void raise() {
		if(!flag_set_) {
			raise_atomic_flag(f_);
			flag_set_ = true;
		}
	}

	inline void drop() {
		if(flag_set_) {
			drop_atomic_flag(f_);
			flag_set_ = false;
		}
	}

private:
	std::atomic_flag& f_;
	bool flag_set_ = false;
};

using Req = link::Req;
using ReqStatus = link::ReqStatus;

// helper struct that contain all status-related bits together
struct status_handle {
	volatile ReqStatus value = ReqStatus::Void;
	std::atomic_flag flag = ATOMIC_FLAG_INIT;
	// mutex is required to implement sleep while Busy status is set
	std::mutex busy_wait;
};

template<typename L, typename F>
static auto link_invoke(
	L* lnk, F f, status_handle& status, bool wait_if_busy = false
) -> decltype(f(lnk)) {
	using ret_t = decltype(f(lnk));

	// make scoped atomic flag and raise it immediately
	auto status_flag = scope_atomic_flag(status.flag);
	// local flag indicating that we have locked busy mutex
	bool busy_waiting = false;

	const auto set_status = [&status, &status_flag, &busy_waiting](ret_t&& res) {
		// set flag depending on result
		status_flag.raise();
		status.value = res ?
			res.value() ?
				ReqStatus::OK :
				ReqStatus::Void :
			ReqStatus::Error;
		// unlock busy mutex if it is locked
		if(busy_waiting) {
			status.busy_wait.unlock();
			busy_waiting = false;
		}
		// and return
		return std::move(res);
	};

	// 1. check Busy state: return error or wait until we can capture the flag in non-Busy state
	// [NOTE] flag is in raised state here
	while(status.value == ReqStatus::Busy) {
		if(!wait_if_busy) return tl::make_unexpected(error::quiet(Error::LinkBusy));
		status_flag.drop();
		const std::lock_guard<std::mutex> wait(status.busy_wait);
		status_flag.raise();
	}
	// here we always have: status is non-busy, flag is raised, busy mutex is unlocked

	// 2. invoke link::f
	try {
		// set Busy status (if status is OK, then we don't lock busy mutex)
		if(status.value == ReqStatus::Void || status.value == ReqStatus::Error) {
			status.value = ReqStatus::Busy;
			status.busy_wait.lock();
			busy_waiting = true;
		}

		// release status flag before possibly long operation
		status_flag.drop();
		// invoke operation, set status and return result
		return set_status(f(lnk));
		// [TODO] we have to notify parent node that content changed
	}
	catch(const error& e) {
		return set_status( tl::make_unexpected(e) );
	}
	catch(const std::exception& e) {
		return set_status( tl::make_unexpected(e.what()) );
	}
	catch(...) {
		return set_status( tl::make_unexpected(error()) );
	}
}

NAMESPACE_END(detail) NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

