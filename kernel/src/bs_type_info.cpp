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

#include "bs_type_info.h"

namespace blue_sky {
	class nil {};
	
	bs_type_info::bs_type_info()
	{
		pinfo_ = &typeid(nil);
		assert(pinfo_);
	}

	bool bs_type_info::is_nil() const {
		return (pinfo_ == &typeid(nil));
	}
}
