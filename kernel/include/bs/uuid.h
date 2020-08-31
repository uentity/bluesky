/// @author uentity
/// @date 08.06.2020
/// @brief Work with UUIDs (generate, parse, etc)
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "atoms.h"
#include "error.h"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_hash.hpp>
#include <boost/uuid/uuid_io.hpp>

NAMESPACE_BEGIN(blue_sky)

using uuid = boost::uuids::uuid;
using uuid_or_err = result_or_err<uuid>;

/// generates random UUID
BS_API auto gen_uuid() -> uuid;

/// non-throwing from string -> UUID conversion
BS_API auto to_uuid(std::string_view s) noexcept -> uuid_or_err;

/// throwing string -> UUID conversion
BS_API auto to_uuid(unsafe_t, std::string_view s) -> uuid;

NAMESPACE_END(blue_sky)
