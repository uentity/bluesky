/**
 * \file shared_array_allocator.cpp
 * \brief
 * \author Sergey Miryanov
 * \date 02.11.2009
 * */
#include "shared_vector.h"
#include "shared_array_allocator.h"

#include <vector>

namespace blue_sky {



  template struct shared_array <uint8_t>;
  template struct shared_array <float16_t>;

  //template struct shared_array_manager <uint8_t>;
  //template struct shared_array_manager <float16_t>;
  //template struct shared_array_manager <int>;

} // namespace blue_sky
