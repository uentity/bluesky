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

#include "bs_array.h"
#include "bs_map.h"

using namespace std;

namespace blue_sky {

BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (int, vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (float, vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (double, vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (std::string, vector_traits));

BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (int, str_val_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (float, str_val_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (double, str_val_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (std::string, str_val_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (sp_obj, str_val_traits));

}	// end of blue_sky namespace

