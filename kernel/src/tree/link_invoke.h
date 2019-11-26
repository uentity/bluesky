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
#include <bs/detail/function_view.h>

#include <atomic>
#include <mutex>
#include <boost/smart_ptr/detail/yield_k.hpp>

NAMESPACE_BEGIN(blue_sky::tree::detail)

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
using status_changed_fh = function_view< void(ReqStatus, ReqStatus) >;
inline constexpr auto nop_status_handler = [](ReqStatus, ReqStatus) {};

// helper struct that contain all status-related bits together
struct status_handle {
	volatile ReqStatus value = ReqStatus::Void;
	std::atomic_flag flag = ATOMIC_FLAG_INIT;
	// mutex is required to implement sleep while Busy status is set
	std::mutex busy_wait;
};

// Invoke `f(lnk)` and properly set status depending on result
// [NOTE] `status_changed_f` is not called when rasing/dropping Busy status
template<typename F>
static auto link_invoke(
	F&& f, status_handle& status, bool wait_if_busy = false,
	status_changed_fh status_changed_f = nop_status_handler
) -> decltype(auto) {
	using R = std::invoke_result_t<F>;
	using ret_t = std::conditional_t< tl::detail::is_expected<R>::value,
		  result_or_err<typename R::value_type>, tl::expected<R, error>
	>;

	// make scoped atomic flag and raise it immediately
	auto status_flag = scope_atomic_flag(status.flag);
	// local flag indicating if we have locked busy mutex
	bool busy_waiting = false;

	// 1. check Busy state: return error or wait until we can capture the flag in non-Busy state
	// [NOTE] flag is in raised state here
	while(status.value == ReqStatus::Busy) {
		if(!wait_if_busy) return ret_t{ error::quiet(Error::LinkBusy) };
		status_flag.drop();
		const std::lock_guard<std::mutex> wait(status.busy_wait);
		status_flag.raise();
	}
	// here we always have: status is non-busy, flag is raised, busy mutex is unlocked

	// setup result processing functor
	auto prev_status = status.value;
	// placeholder for return value
	ret_t res;

	const auto set_status =
	[&status, &status_flag, &busy_waiting, prev_status, status_changed_f](ret_t&& res) {
		// set flag depending on result
		status_flag.raise();
		status.value = res ?
			res.value() ?
				ReqStatus::OK :
				ReqStatus::Void :
			ReqStatus::Error;
		// [NOTE] capture new & old status values atomically while flag is raised
		auto finally = scope_guard{[new_status = status.value, prev_status, status_changed_f] {
			status_changed_f(new_status, prev_status);
		}};
		// unlock busy mutex if it is locked
		if(busy_waiting) {
			status.busy_wait.unlock();
			busy_waiting = false;
		}
		// and return
		// [NOTE] explicity drop flag, because status callback can take long time
		status_flag.drop();
		return std::move(res);
	};

	// 2. invoke link::f
	auto er = error::eval_safe([&] {
		// set Busy status (if status is OK, then we don't lock busy mutex)
		if(status.value == ReqStatus::Void || status.value == ReqStatus::Error) {
			status.value = ReqStatus::Busy;
			status.busy_wait.lock();
			busy_waiting = true;
		}

		// release status flag before possibly long operation
		status_flag.drop();
		// invoke operation and capture return value
		res = std::invoke(std::forward<F>(f));
	});
	if(er) res = tl::make_unexpected(std::move(er));
	return set_status(std::move(res));
}

NAMESPACE_END(blue_sky::tree::detail)
