/**
 * \file shared_array_allocator.cpp
 * \brief
 * \author Sergey Miryanov
 * \date 02.11.2009
 * */
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

