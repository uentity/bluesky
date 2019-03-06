/// @file
/// @author uentity
/// @date 21.03.2017
/// @brief Include complete BS API
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "log.h"
#include "assert.h"
// errors handling
#include "error.h"
#include "throw_exception.h"
#include "tree/errors.h"
// kernel API
#include "any_array.h"
#include "kernel/kernel.h"
// other things
#include "misc.h"

// Python-related dclarations
#if defined(BSPY_EXPORTING) || defined(BSPY_EXPORTING_PLUGIN)
#include "python/common.h"
#endif

