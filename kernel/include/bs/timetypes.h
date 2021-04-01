/// @file
/// @author uentity
/// @date 19.03.2019
/// @brief Time-related BS types (timestamp, timespan)
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "setup_common_api.h"
#include <caf/timestamp.hpp>

#include <iosfwd>

namespace blue_sky {

using timespan = caf::timespan;
using timestamp = caf::timestamp;

/// denote infinite duration
inline constexpr auto infinite = caf::infinite;

/// get now timestamp from sys clock
BS_API timestamp make_timestamp();

/// formatting support
BS_API std::string to_string(timespan t);
BS_API std::string to_string(timestamp t);
BS_API std::ostream& operator <<(std::ostream& os, timestamp t);
BS_API std::ostream& operator <<(std::ostream& os, timespan t);

} // eof blue_sky namespace
