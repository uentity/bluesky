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

/**
 * \file smart_ptr.cpp
 * \brief
 * \author Sergey Miryanov
 * \date 22.08.2008
 * */

#include <cassert>
#include "setup_common_api.h"
//#include "bs_refcounter.h"
#include "bs_object_base.h"
#include "bs_link.h"

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

} // namespace blue_sky
