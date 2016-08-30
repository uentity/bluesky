/// @file
/// @author uentity
/// @date 24.08.2016
/// @brief BlueSky kernel_impl definition
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel.h>
#include "kernel_logging_subsyst.h"

namespace blue_sky {

class kernel::kernel_impl : public kernel_logging_subsyst {
public:
	static spdlog::logger& get_log(const char* name) {
		// let's create C++11 thread-safe singleton
		// static variable should be initialized only in one thread!
		// we need to use VS2015 to support this
		static kernel_logging_subsyst klog;

		return klog.get_log(name);
	}
};

} /* namespace blue_sky */

