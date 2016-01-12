/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Contains BlueSky storage tables implimentations
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

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

