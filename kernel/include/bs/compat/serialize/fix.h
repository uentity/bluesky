/// @file
/// @author uentity
/// @date 28.10.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "fixdata.h"
#include "fixint.h"
#include "fixreal.h"
#include "fixstr.h"
#include "fixcont.h"

// specify chain of data fixers during serialization in BlueSky
namespace blue_sky {

template< class Archive >
struct serialize_first_fixer< serialize_fix_data< Archive > > {
	typedef
	serialize_fix_cont <
		serialize_fix_int <
			serialize_fix_real<
				serialize_fix_wstring< >
			>
		>
	>
	type;
};

}

