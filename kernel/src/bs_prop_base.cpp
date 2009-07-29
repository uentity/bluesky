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
	template< template< class > class table_traits >
	data_table< table_traits >::data_table(bs_type_ctor_param /*param*/)
	{}

	//copy ctor
	template< template< class > class table_traits >
	data_table< table_traits >::data_table(const data_table< table_traits >& src) : bs_refcounter (src), objbase ()
	{
		*this = src;
	}

	BLUE_SKY_TYPE_STD_CREATE_T(data_table< str_val_table >);
	BLUE_SKY_TYPE_STD_CREATE_T(data_table< bs_array >);
	BLUE_SKY_TYPE_STD_COPY_T(data_table< str_val_table >);
	BLUE_SKY_TYPE_STD_COPY_T(data_table< bs_array >);

	BLUE_SKY_TYPE_IMPL_T(data_table< str_val_table >, objbase, "str_data_table",
		"Table of values of mixed types addressed by string key", "");
	BLUE_SKY_TYPE_IMPL_T(data_table< bs_array >, objbase, "idx_data_table",
		"Table of values of mixed types addressed by index", "");

	BS_TYPE_IMPL_T_MEM(str_val_table, std::string)
	BS_TYPE_IMPL_T_MEM(str_val_table, sp_obj)
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

