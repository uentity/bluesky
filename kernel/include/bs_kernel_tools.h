/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Some useful tools to debug BlueSky kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _BS_KERNEL_TOOLS_H
#define _BS_KERNEL_TOOLS_H

#include "bs_kernel.h"

namespace blue_sky {

class BS_API kernel_tools {
public:

	static std::string print_loaded_types();

	static std::string walk_tree(bool silent = false);

	static std::string print_registered_instances();

	static std::string get_backtrace (int backtrace_depth = 16);
};

}	// blue_sky namespace

#endif

