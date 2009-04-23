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

#include "bs_arrbase.h"
#include "bs_array.h"

using namespace std;

namespace blue_sky {

BS_TYPE_IMPL_T_EXT_MEM(bs_array_t, 2, (int, vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array_t, 2, (float, vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array_t, 2, (double, vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array_t, 2, (std::string, vector_traits));

}	// end of blue_sky namespace

