/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Common parts of BlueSky
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/common.h>
#include <bs/type_descriptor.h>

using namespace std;

#define BS_NIL_TYPE_TAG "__blue_sky_nil_type__"

namespace blue_sky {

//------------------------------type_descriptor-------------------------------------------------------------

bool upcastable_eq::operator()(const type_descriptor& td1, const type_descriptor& td2) const {
	if(td1 == td2) return true;

	const type_descriptor* cur_td = &td2.parent_td();
	while(!cur_td->is_nil()) {
		if(td1 == *cur_td)
			return true;
		cur_td = &cur_td->parent_td();
	}
	return false;
}

// obtain Nil type_descriptor instance
const type_descriptor& type_descriptor::nil() {
	static type_descriptor nil_td(BS_NIL_TYPE_TAG);
	return nil_td;
}

}  // eof blue_sky namespace

