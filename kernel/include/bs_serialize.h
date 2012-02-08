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

#ifndef BS_SERIALIZE_MIZAXRNW
#define BS_SERIALIZE_MIZAXRNW

#include "bs_common.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/archive/polymorphic_iarchive.hpp>
#include <boost/archive/polymorphic_oarchive.hpp>

#include "bs_serialize_macro.h"
#include "bs_serialize_decl.h"
#include "bs_serialize_overl.h"
#include "bs_serialize_text.h"

#include "smart_ptr_serialize.h"
#include "bs_array_serialize.h"

// add empty serialize fcn for objbase
#include "bs_object_base.h"
//BLUE_SKY_CLASS_SRZ_FCN_DECL(serialize, blue_sky::objbase)

BLUE_SKY_TYPE_SERIALIZE_GUID(blue_sky::objbase)

#endif /* end of include guard: BS_SERIALIZE_MIZAXRNW */

