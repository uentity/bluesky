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

#include "bs_cube_t.h"
#include "bs_kernel.h"
#include "bs_prop_base.h"
#include <iostream>

#define DUMB(A, B) BOOST_PP_SEQ_ENUM(A), BOOST_PP_SEQ_ENUM(B)

#define SRC (class)(template< class > class)
#define TLIST BS_TLIST_FORMER(SRC)
#define CLIST BS_CLIST_FORMER(SRC)
#define RES(T, t_params) (template< BS_TLIST_FORMER(t_params) > BS_API_PLUGIN) //, (T< BS_CLIST_FORMER(t_params) >)
#define EXPAND_RES RES(bs_cube_tt, SRC)
#define EXP_RES_NOBR BOOST_PP_TUPLE_REM_CTOR(2, EXPAND_RES)

namespace blue_sky {
BS_TYPE_IMPL_T_MEM(blue_sky::bs_array, int)
BS_TYPE_IMPL_T_MEM(str_val_table, smart_ptr< bs_cube_t< float > >)
BS_TYPE_IMPL_T_MEM(str_val_table, smart_ptr< bs_cube_t< int > >)
}

using namespace blue_sky;
using namespace std;

//template version
template< typename T >
bs_cube_t< T >::bs_cube_t(bs_type_ctor_param /*param*/)
{}

template< typename T >
bs_cube_t< T >::bs_cube_t(const bs_cube_t< T >& src)
{
	*this = src;
}

template< typename T >
bs_cube_t< T >::~bs_cube_t()
{}

template< typename T >
void bs_cube_t< T >::test()
{
	//STAGE1
	//TLIST
	//CLIST
	//EXPAND_RES
	//EXP_RES_NOBR
	//cout << BOOST_PP_STRINGIZE(STAGE1) << BOOST_PP_STRINGIZE(TLIST) << BOOST_PP_STRINGIZE(CLIST) << endl;
	smart_ptr< bs_array< int > > p_ivt = give_kernel::Instance().create_object(blue_sky::bs_array< int >::bs_type());
	var_ = 0;
}

//------------- bs_cube_tt ---------------------------------------------------------------------------------------------
template< class T, template< class > class U >
bs_cube_tt< T, U >::bs_cube_tt(bs_type_ctor_param param)
{
	smart_ptr< str_data_table > sp_dt(param, bs_dynamic_cast());
	if(param) {
		smart_ptr< u_type > pu = sp_dt->extract_value< smart_ptr< u_type > >("U");
		if(pu) u_ = *pu;
	}
}

template< class T, template< class > class U >
typename bs_cube_tt< T, U >::sp_cube_tt bs_cube_tt< T, U >::create(const u_type& u) {
	lsmart_ptr< smart_ptr< str_data_table > > sp_dt(BS_KERNEL.create_object(str_data_table::bs_type(), true));
	sp_dt->add_item< smart_ptr< u_type > >("U", smart_ptr< u_type >(&u));
	return sp_cube_tt(BS_KERNEL.create_object(bs_type(), false, sp_dt));
}

//template< class T, template< class > class U >
//bs_cube_tt< T, U >::bs_cube_tt(const bs_cube_tt& src)
//{
//	//*this = src;
//}

template< class T, template< class > class U >
void bs_cube_tt< T, U >::test()
{
	u_.test();
}

namespace blue_sky {

	BLUE_SKY_TYPE_STD_CREATE_T_DEF(bs_cube_t, (class))
//	BLUE_SKY_TYPE_STD_CREATE_T(bs_cube_t< int >);
//	BLUE_SKY_TYPE_STD_CREATE_T(bs_cube_t< float >);
//	BLUE_SKY_TYPE_STD_CREATE_T(bs_cube_t< double >);

	BLUE_SKY_TYPE_STD_COPY_T_DEF(bs_cube_t, (class))
//	BLUE_SKY_TYPE_STD_COPY_T(bs_cube_t< int >);
//	BLUE_SKY_TYPE_STD_COPY_T(bs_cube_t< float >);
//	BLUE_SKY_TYPE_STD_COPY_T(bs_cube_t< double >);

	//BLUE_SKY_TYPE_IMPL_T(bs_cube_t< int >, objbase, "int_bs_cube_t", "Short test int_bs_cube_t description", "")
	BLUE_SKY_TYPE_IMPL_T_SHORT(bs_cube_t< int >, objbase, "Short test int_bs_cube_t description")
	BLUE_SKY_TYPE_IMPL_T_SHORT(bs_cube_t< float >, objbase, "Short test float_bs_cube_t description")
	BLUE_SKY_TYPE_IMPL_T_SHORT(bs_cube_t< double >, objbase, "Short test double_bs_cube_t description")

	//------bs_cube_tt
	BLUE_SKY_TYPE_STD_CREATE_T_DEF(bs_cube_tt, (class)(template< class > class))

	BLUE_SKY_TYPE_IMPL_T_EXT(2, (bs_cube_tt< int, bs_cube_t >), 1, (objbase), "bs_cube_tt< int >", "Short test int_bs_cube_tt description", "", true)
	BLUE_SKY_TYPE_IMPL_T_NOCOPY_SHORT(bs_cube_tt< float >, objbase, "Short test float_bs_cube_tt description")
}
