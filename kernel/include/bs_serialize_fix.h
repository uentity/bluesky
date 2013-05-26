// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

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

