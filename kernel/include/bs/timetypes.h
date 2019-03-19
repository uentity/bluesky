/// @file
/// @author uentity
/// @date 19.03.2019
/// @brief Time-related BS types (timestamp, timespan)
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "common.h"
#include <caf/timestamp.hpp>

#include <iosfwd>

namespace blue_sky {

using timespan = caf::timespan;
using timestamp = caf::timestamp;

using caf::make_timestamp;

/// formatting support
// [NOTE] accept by value, because objects are small
BS_API std::string to_string(timespan t);
BS_API std::string to_string(timestamp t);
BS_API std::ostream& operator <<(std::ostream& os, timestamp t);
BS_API std::ostream& operator <<(std::ostream& os, timespan t);

} // eof blue_sky namespace
