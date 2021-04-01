/// @author Alexander Gagarin (@uentity)
/// @date 03.04.2021
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "atoms.h"
#include "fwd.h"
#include "common.h"
#include "error.h"
#include "uuid.h"

CAF_BEGIN_TYPE_ID_BLOCK(bs, blue_sky::detail::bs_cid_begin)

	CAF_ADD_TYPE_ID(bs, (std::vector<std::string>))
	CAF_ADD_TYPE_ID(bs, (std::vector<std::size_t>))

	CAF_ADD_TYPE_ID(bs, (blue_sky::error::box))
	CAF_ADD_TYPE_ID(bs, (std::vector<blue_sky::error::box>))

	CAF_ADD_TYPE_ID(bs, (blue_sky::sp_obj))
	CAF_ADD_TYPE_ID(bs, (blue_sky::sp_cobj))
	CAF_ADD_TYPE_ID(bs, (blue_sky::sp_objnode))
	CAF_ADD_TYPE_ID(bs, (blue_sky::sp_cobjnode))

	CAF_ADD_TYPE_ID(bs, (blue_sky::uuid))

CAF_END_TYPE_ID_BLOCK(bs)
