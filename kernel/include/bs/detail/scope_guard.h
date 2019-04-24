/// @file
/// @author uentity
/// @date 15.08.2018
/// @brief Lightweight scope guard that allows executing custom code in destructor
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

// [NOTE] Code is actually copy-paste from corresponding CAF sources

#include <utility>

namespace blue_sky { namespace detail {

/// A lightweight scope guard implementation.
template <class Fun>
class scope_guard {
	scope_guard() = delete;
	scope_guard(const scope_guard&) = delete;
	scope_guard& operator=(const scope_guard&) = delete;

public:
	scope_guard(Fun f) : fun_(std::move(f)), enabled_(true) {}

	scope_guard(scope_guard&& other)
		: fun_(std::move(other.fun_)), enabled_(other.enabled_)
	{
		other.enabled_ = false;
	}

	~scope_guard() {
		if(enabled_) fun_();
	}

	/// Disables this guard, i.e., the guard does not
	/// run its cleanup code as it goes out of scope.
	inline void disable() { enabled_ = false; }

private:
	Fun fun_;
	bool enabled_;
};

/// Creates a guard that executes `f` as soon as it goes out of scope.
/// @relates scope_guard
template <class Fun>
auto make_scope_guard(Fun f) -> scope_guard<Fun> {
	return {std::move(f)};
}

} /* namespace blue_sky::detail */

// import make_scope_guard into blue_sky namespace
using detail::make_scope_guard;
using detail::scope_guard;

} /* namespace blue_sky */

