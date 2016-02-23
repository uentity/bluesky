/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Implimentations for combase class
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

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
