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

	// put some common specializations of bs_array to kernel lib
	BS_TYPE_IMPL_T_MEM(bs_array, float);
	BS_TYPE_IMPL_T_MEM(bs_array, double);
	BS_TYPE_IMPL_T_MEM(bs_array, int);
	BS_TYPE_IMPL_T_MEM(bs_array, ulong);
	BS_TYPE_IMPL_T_MEM(bs_array, unsigned char);
	BS_TYPE_IMPL_T_MEM(bs_array, std::string);

	// data-table related exports
	BLUE_SKY_TYPE_STD_CREATE_T(data_table< str_val_table >);
	BLUE_SKY_TYPE_STD_CREATE_T(data_table< bs_array >);
	BLUE_SKY_TYPE_STD_COPY_T(data_table< str_val_table >);
	BLUE_SKY_TYPE_STD_COPY_T(data_table< bs_array >);

	BLUE_SKY_TYPE_IMPL_T(data_table< str_val_table >, objbase, "str_data_table",
		"Table of values of mixed types addressed by string key", "");
	BLUE_SKY_TYPE_IMPL_T(data_table< bs_array >, objbase, "idx_data_table",
		"Table of values of mixed types addressed by index", "");

	BS_TYPE_IMPL_T_MEM(str_val_table, std::string);
	BS_TYPE_IMPL_T_MEM(str_val_table, sp_obj);

	kernel::types_enum register_bs_array() {
		kernel::types_enum te;
		te.push_back(bs_array< float >::bs_type());
		te.push_back(bs_array< double >::bs_type());
		te.push_back(bs_array< int >::bs_type());
		te.push_back(bs_array< ulong >::bs_type());
		te.push_back(bs_array< unsigned char >::bs_type());
		te.push_back(bs_array< std::string >::bs_type());
		return te;
	}

	kernel::types_enum register_data_table() {
		kernel::types_enum te;
		te.push_back(data_table< str_val_table >::bs_type());
		te.push_back(data_table< bs_array >::bs_type());
		te.push_back(str_val_table< std::string >::bs_type());
		te.push_back(str_val_table< sp_obj >::bs_type());
		return te;
	}
}

