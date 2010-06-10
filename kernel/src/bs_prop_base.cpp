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

/*!
 * \file bs_prop_base.cpp
 * \brief Contains blue-sky storage tables implimentations.
 * \author uentity
 */
#include "bs_prop_base.h"

//DEBUG!
//#include <iostream>

using namespace std;

namespace blue_sky {
	//default ctor
	template< template< class, template< class > class > class table_t, template< class > class cont_traits >
	data_table< table_t, cont_traits >::data_table(bs_type_ctor_param param)
		: bs_refcounter(), objbase(param)
	{}

	//copy ctor
	template< template< class, template< class > class > class table_t, template< class > class cont_traits >
	data_table< table_t, cont_traits >::data_table(const data_table< table_t, cont_traits >& src) 
		: bs_refcounter(), objbase(src)
	{
		*this = src;
	}

	BLUE_SKY_TYPE_STD_CREATE_T(str_data_table);
	BLUE_SKY_TYPE_STD_CREATE_T(idx_data_table);
	BLUE_SKY_TYPE_STD_COPY_T(str_data_table);
	BLUE_SKY_TYPE_STD_COPY_T(idx_data_table);

	// data-table related exports

	BLUE_SKY_TYPE_IMPL_T_EXT(2, (data_table< bs_map, str_val_traits >), 1, (objbase), "str_data_table",
		"Table of values of mixed types addressed by string key", "", false);
	BLUE_SKY_TYPE_IMPL_T_EXT(2, (data_table< bs_array, vector_traits >), 1, (objbase), "idx_data_table",
		"Table of values of mixed types addressed by index", "", false);

	kernel::types_enum register_data_table() {
		kernel::types_enum te;
		te.push_back(data_table< bs_map, str_val_traits >::bs_type());
		te.push_back(data_table< bs_array, vector_traits >::bs_type());
		return te;
	}
}

//using namespace blue_sky;

//template class str_val_table< int >;
//template class str_val_table< float >;
//template class str_val_table< double >;
//template class str_val_table< bool >;
//template class str_val_table< std::string >;
//#if !defined(_WIN32) && !defined(_MSC_VER)
//template class str_val_table< objbase >;
//template class str_val_table< combase >;
//#else
//template class str_val_table< smart_ptr< objbase, true > >;
//template class str_val_table< smart_ptr< combase, true > >;
//#endif


// template class bs_array< int >;
// template class bs_array< float >;
// template class bs_array< double >;
// template class bs_array< bool >;
// template class bs_array< std::string >;
// #if !defined(_WIN32) && !defined(_MSC_VER)
// template class bs_array< objbase >;
// template class bs_array< combase >;
// #else
// template class bs_array< smart_ptr< objbase, true > >;
// template class bs_array< smart_ptr< combase, true > >;
// #endif

