/// @file
/// @author Sergey Miryanov
/// @date 02.11.2009
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#if defined(BSPY_EXPORTING) && defined(UNIX)
#include <boost/python/detail/wrap_python.hpp>
#endif
#include "shared_vector.h"

#include <vector>

namespace blue_sky {
namespace private_ {

  template struct shared_array <uint8_t>;
  template struct shared_array <float16_t>;

} // namespace private_
} // namespace blue_sky

