/// @file
/// @author uentity
/// @date 06.11.2009
/// @brief Sergey Miryanov
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_SHARED_ARRAY_DETAIL_H_
#define BS_SHARED_ARRAY_DETAIL_H_

#include "throw_exception.h"

namespace blue_sky {
namespace detail {

  template <typename pointer, typename size_type, typename allocator_t>
  void
  deallocate (pointer p, size_type n, allocator_t &allocator)
  {
    if (p)
      {
        allocator.deallocate (p, n);
      }
  }

  template <bool is_arithmetic>
  struct detail_t
  {
  };

  template <>
  struct detail_t <false>
  {
    template <typename forward_iterator, typename allocator_t>
    static void
    destroy (forward_iterator first, forward_iterator last, allocator_t &allocator)
    {
      for (; first != last; ++first)
        allocator.destroy (&*first);
    }
  };

  template <>
  struct detail_t <true>
  {
    template <typename forward_iterator, typename allocator_t>
    static void
    destroy (forward_iterator /*first*/, forward_iterator /*last*/, allocator_t & /*allocator*/)
    {
    }
  };

} // namespace detail
} // namespace blue_sky

#endif // #ifndef BS_SHARED_ARRAY_DETAIL_H_

