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
#ifdef BSPY_EXPORTING_PLUGIN
#include <boost/python.hpp>
#endif

#include "bs_abstract_storage.h"
#include "bs_command.h"
#include "bs_kernel.h"

namespace blue_sky { 
empty_storage::empty_storage(bs_type_ctor_param /*param*/)
{}

empty_storage::empty_storage(const empty_storage& src) : bs_refcounter (src), bs_abstract_storage ()
{
	*this = src;
}

BLUE_SKY_TYPE_STD_CREATE(empty_storage);
BLUE_SKY_TYPE_STD_COPY(empty_storage);
BLUE_SKY_TYPE_IMPL(empty_storage, objbase, "empty_storage", "", "");

} 	//end of blue-sky namespace

