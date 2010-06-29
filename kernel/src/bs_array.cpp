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
// bs_array
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (int, vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (float, vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (double, vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (std::string, vector_traits));

BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (int, shared_array_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (float, shared_array_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (double, shared_array_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (std::string, shared_array_traits));

BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (int, shared_vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (float, shared_vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (double, shared_vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (std::string, shared_vector_traits));

// bs_map
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (int, str_val_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (float, str_val_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (double, str_val_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (std::string, str_val_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (sp_obj, str_val_traits));

kernel::types_enum register_bs_array() {
	kernel::types_enum te;
	te.push_back(bs_array< int, vector_traits >::bs_type());
	te.push_back(bs_array< float, vector_traits >::bs_type());
	te.push_back(bs_array< double, vector_traits >::bs_type());
	te.push_back(bs_array< std::string, vector_traits >::bs_type());

	te.push_back(bs_array< int, shared_array_traits >::bs_type());
	te.push_back(bs_array< float, shared_array_traits >::bs_type());
	te.push_back(bs_array< double, shared_array_traits >::bs_type());
	te.push_back(bs_array< std::string, shared_array_traits >::bs_type());

	te.push_back(bs_array< int, shared_vector_traits >::bs_type());
	te.push_back(bs_array< float, shared_vector_traits >::bs_type());
	te.push_back(bs_array< double, shared_vector_traits >::bs_type());
	te.push_back(bs_array< std::string, shared_vector_traits >::bs_type());

	te.push_back(bs_map< int, str_val_traits >::bs_type());
	te.push_back(bs_map< float, str_val_traits >::bs_type());
	te.push_back(bs_map< double, str_val_traits >::bs_type());
	te.push_back(bs_map< std::string, str_val_traits >::bs_type());
	te.push_back(bs_map< sp_obj, str_val_traits >::bs_type());
	return te;
}

}	// end of blue_sky namespace

