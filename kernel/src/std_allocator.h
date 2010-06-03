/**
 * \file std_allocator.h
 * \brief wrap new and delete ops to use as an allocator in bs_bos_core::memory_manager
 * \author Sergey Miryanov
 * \date 11.06.2009
 * */
#ifndef BS_BOS_CORE_BASE_STD_ALLOCATOR_H_
#define BS_BOS_CORE_BASE_STD_ALLOCATOR_H_

namespace blue_sky {
namespace detail {

  struct std_allocator : memory_manager::allocator_interface
  {
    void *
    allocate (size_t size)
    {
      count_++;
      return malloc (size);
    }

    void
    deallocate (void *ptr)
    {
      count_--;
      free (ptr);
    }

    static std_allocator &
    instance ()
    {
      static std_allocator *sa = 0;
      if (!sa)
        {
          sa = new std_allocator ();
        }

      return *sa;
    }
  };


} // namespace detail

typedef detail::std_allocator allocator_t;

} // namespce blue_sky


#endif // #ifndef BS_BOS_CORE_BASE_STD_ALLOCATOR_H_

