/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_SERIALIZE_FIX_NC6BGKHP
#define BS_SERIALIZE_FIX_NC6BGKHP

#include "bs_serialize_fixdata.h"
#include "bs_serialize_fixreal.h"
#include "bs_serialize_fixstr.h"
#include "bs_serialize_fixcont.h"

// specify chain of data fixers during serialization in BlueSky
namespace blue_sky {

template< class Archive >
struct serialize_first_fixer< serialize_fix_data< Archive > > {
	typedef
	serialize_fix_cont <
		serialize_fix_real<
			serialize_fix_wstring< >
		>
	>
	type;
};

}

#endif /* end of include guard: BS_SERIALIZE_FIX_NC6BGKHP */

