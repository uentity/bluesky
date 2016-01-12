/// @file
/// @author Sergey Miryanov
/// @date 15.06.2009
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_BOS_CORE_BASE_HOARD_ALLOCATOR_H_
#define BS_BOS_CORE_BASE_HOARD_ALLOCATOR_H_

#include "libhoard.h"

namespace blue_sky {
namespace detail {

    struct hoard_allocator : memory_manager::allocator_interface
    {
      hoard_allocator ()
      {
      }

      void *
      allocate (size_t size)
      {
        count_++;
        void *ptr = hoard_malloc (size);
        return ptr;
      }

      void
      deallocate (void *ptr)
      {
        count_--;
        hoard_free (ptr);
      }

      static hoard_allocator &
      instance ()
      {
        static hoard_allocator *hoard_ = 0;
        if (!hoard_)
          {
            hoard_ = new hoard_allocator ();
          }

        return *hoard_;
      }
    };


} // namespace detail

typedef detail::hoard_allocator allocator_t;

} // namespace blue_sky

#endif // #ifndef BS_BOS_CORE_BASE_HOARD_ALLOCATOR_H_

