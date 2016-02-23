/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "setup_common_api.h"
//#include "bs_refcounter.h"
#include "bs_object_base.h"
#include "bs_link.h"

#include <cassert>

namespace blue_sky {
void BS_API bs_refcounter_add_ref (const bs_refcounter *p) {
	//assert (p);
	if (p)
		p->add_ref ();
}

void BS_API bs_refcounter_del_ref (const bs_refcounter *p) {
	//assert (p);
	if (p) {
		if(p->refs() == 1) {
			// call on_delete signal for objbase instances
			// if p is to be disposed
			sp_obj o(p, bs_dynamic_cast());
			if(o)
				o->fire_signal(objbase::on_delete, sp_obj(NULL));
		}
		// decrement reference counter
		p->del_ref ();
	}
}

void BS_API usual_deleter_unsafe::operator()(void *p) const {
	delete static_cast <char *> (p);
}

void BS_API array_deleter_unsafe::operator()(void *p) const {
	delete [] static_cast <char *> (p);
}

void BS_API bs_obj_deleter_unsafe::operator()(void *p) const {
	static_cast< bs_refcounter const* >(p)->del_ref();
}

} // namespace blue_sky
