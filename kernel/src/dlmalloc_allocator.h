/**
 * \file dlmalloc_allocator.h
 * \brief
 * \author Sergey Miryanov
 * \date 06.07.2009
 * */
#ifndef BS_BOS_CORE_DLMALLOC_ALLOCATOR_H_
#define BS_BOS_CORE_DLMALLOC_ALLOCATOR_H_

extern "C" {
#include "dlmalloc.h"
}

#define USE_DL_PREFIX

namespace blue_sky {
namespace detail {

  struct dlmalloc_allocator : memory_manager::allocator_interface
  {
    dlmalloc_allocator ()
    {
    }

    void *
    allocate (size_t size)
    {
      count_++;
      void *ptr = dlmalloc_x (size);
      return ptr;
    }

    void
    deallocate (void *ptr)
    {
      count_--;
      dlfree_x (ptr);
    }

    static dlmalloc_allocator &
    instance ()
    {
      static dlmalloc_allocator *dlmalloc_ = new dlmalloc_allocator ();
      return *dlmalloc_;
    }
  };

} // namespace detail

typedef detail::dlmalloc_allocator allocator_t;

} // namespace blue_sky

#endif // #ifndef BS_BOS_CORE_DLMALLOC_ALLOCATOR_H_

