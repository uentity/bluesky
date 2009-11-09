/**
 * \file shared_array_allocator.cpp
 * \brief
 * \author Sergey Miryanov
 * \date 02.11.2009
 * */
#include "shared_vector.h"
#include "shared_array_allocator.h"

namespace blue_sky {

  template struct shared_array <uint8_t>;
  template struct shared_array <float16_t>;

} // namespace blue_sky

