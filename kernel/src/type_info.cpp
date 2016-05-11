/// @file
/// @author uentity
/// @date 28.04.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/type_info.h>

namespace blue_sky {

class nil {};

std::type_index nil_type_info() {
	return BS_GET_TI(nil);
}

bool is_nil(const std::type_index& t) {
	return t == std::type_index(typeid(nil));
}

} // eof blue_sky namespace

