/// @file
/// @author uentity
/// @date 23.08.2016
/// @brief BlueSky kernel implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel.h>

namespace blue_sky {

class kernel::kernel_impl {
};

kernel::kernel() : pimpl_(new kernel_impl) {}

kernel::~kernel() {}

void kernel::init() {}

void kernel::cleanup() {}

} /* namespace blue_sky */

