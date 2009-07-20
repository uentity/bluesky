/**
 * \file tlsf_allocator.h
 * \brief wrapper to adapt TLSF allocator to use as an allocator in bs_bos_core::memory_manager
 * \author Sergey Miryanov
 * \date 11.06.2009
 * */
#ifndef BS_BOS_CORE_BASE_TLSF_ALLOCATOR_H_
#define BS_BOS_CORE_BASE_TLSF_ALLOCATOR_H

extern "C" {
#include "tlsf.h"
}

namespace blue_sky {
namespace detail {

  struct tlsf_allocator : memory_manager::allocator_interface
  {
    tlsf_allocator (size_t pool_size)
    : pool_size_ (pool_size)
    {
      void *initial_pool = new char [pool_size_];
      memory_list_.push_back (initial_pool);

      if (-1 == init_memory_pool (pool_size_, initial_pool))
        throw bs_exception ("tlsf::tlsf", "Can't initialize memory pool");
    }

    ~tlsf_allocator ()
    {
      //destroy_memory_poll ();
    }

    void *
    allocate (size_t size)
    {
      void *ptr = tlsf_malloc (size);
      if (!ptr)
        {
          if (size > pool_size_)
            {
              pool_size_ = size * 2;
            }
          void *pool = new char [pool_size_];
          memory_list_.push_back (pool);
          add_new_area (pool, pool_size_, memory_list_.front ());

          ptr = tlsf_malloc (size);
#ifdef BS_BOS_CORE_DEBUG_MEMORY
          if (!ptr)
            {
              // TLSF statistic adds 0.10s to 0.41s if enabled (spe7/1a model)
              for (size_t i = 0, cnt = memory_list_.size (); i < cnt; ++i)
                {
                  BSOUT << "tlsf.used_size: " << get_used_size (memory_list_[i]) << bs_end;
                  BSOUT << "tlsf.max_size: " << get_max_size (memory_list_[i]) << bs_end;
                }
            }
#endif
        }

      count_++;
      return ptr;
    }

    void 
    deallocate (void *ptr)
    {
      count_--;
      tlsf_free (ptr);
    }

    size_t                pool_size_;
    std::vector <void *>  memory_list_;


    static tlsf_allocator &
    instance ()
    {
      size_t pool_size = 20 * 1024 * 1024;
      static tlsf_allocator *tlsf_ = 0;
      if (!tlsf_)
        {
          tlsf_ = new tlsf_allocator (pool_size);
        }

      return *tlsf_;
    }
  };

} // namespace detail


typedef detail::tlsf_allocator allocator_t;

} // namespace blue_sky


#endif // #ifndef BS_BOS_CORE_BASE_TLSF_ALLOCATOR_H_
