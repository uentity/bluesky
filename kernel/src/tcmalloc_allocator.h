/// @file
/// @author Sergey Miryanov
/// @date 06.07.2009
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_BOS_CORE_TCMALLOC_ALLOCATOR_H_
#define BS_BOS_CORE_TCMALLOC_ALLOCATOR_H_

extern "C" {
#include "tcmalloc.h"
}

#define USE_DL_PREFIX

namespace blue_sky {
namespace detail {

  struct tcmalloc_allocator : memory_manager::allocator_interface
  {
    tcmalloc_allocator ()
    {
    }

    void *
    allocate (size_t size)
    {
      count_++;
      void *ptr = tcmalloc_malloc (size);
      return ptr;
    }

    void
    deallocate (void *ptr)
    {
      count_--;
      tcmalloc_free (ptr);
    }

    static tcmalloc_allocator &
    instance ()
    {
      static tcmalloc_allocator *tcmalloc_ = new tcmalloc_allocator ();
      return *tcmalloc_;
    }
  };

} // namespace detail

typedef detail::tcmalloc_allocator allocator_t;

} // namespace blue_sky

#endif // #ifndef BS_BOS_CORE_TCMALLOC_ALLOCATOR_H_

