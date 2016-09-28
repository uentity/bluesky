/// @file
/// @author uentity
/// @date 28.09.2016
/// @brief wrapper_kernel struct declaration
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef WRAPPER_KERNEL_H
#define WRAPPER_KERNEL_H

#include "bs_kernel.h"

namespace blue_sky { namespace bs_private {

/// @brief Wrapper allowing to do some initialization on first give_kernel()::Instance() call
/// just after the kernel class is created
struct wrapper_kernel {
	kernel k_;

	kernel& (wrapper_kernel::*ref_fun_)();

	static void kernel_cleanup();

	// constructor
	wrapper_kernel();
	// destructor
	~wrapper_kernel();

	// normal getter - just returns kernel reference
	kernel& usual_kernel_getter();

	// when kernel reference is obtained for the first time
	kernel& initial_kernel_getter();

	kernel& k_ref();

#ifdef BSPY_EXPORTING
	// obtain kernel's Python module
	PyObject* k_py_module();
#endif
};

}} /* namespace blue_sky */

#endif /* end of include guard: WRAPPER_KERNEL_H */
