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
 * \file bs_command.cpp
 * \brief Implimentations for combase class
 * \author uentity
 */
#include "bs_command.h"

using namespace blue_sky;

////! The empty combase constructor.
//combase::combase(bs_type_ctor_param param)
//{}
//
////copy constructor
//combase::combase(const combase& src)
//: objbase(src)
//{}
//
//BS_TYPE_IMPL_COMMON(combase, objbase)

bool combase::can_unexecute() const
{
	return true;
}

//void combase::dispose() const {
//	delete this;
//}

//namespace blue_sky {
//BS_TYPE_IMPL_T_DEF(blue_sky::str_val_table, sp_obj)
//BS_TYPE_IMPL_T_DEF(blue_sky::str_val_table, int)
//}
