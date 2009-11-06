/**
 * \file aligned_allocator.h
 * \brief aligned allocator for stl containers 
 * \author Sergey Miryanov
 * \date 11.01.2009
 * */
#ifndef BS_TOOLS_ALIGNED_ALLOCATOR_H_
#define BS_TOOLS_ALIGNED_ALLOCATOR_H_

#include "memory_manager.h"
#include "bs_kernel.h"

namespace blue_sky
{
  template <typename T, size_t alignment_ = 16>
  struct aligned_allocator : public std::allocator <T>
  {
    typedef std::allocator <T>                base_t;
    typedef aligned_allocator <T, alignment_> this_t;
    typedef typename base_t::size_type        size_type;
    typedef typename base_t::difference_type  difference_type;
    typedef typename base_t::pointer          pointer;
    typedef typename base_t::const_pointer    const_pointer;
    typedef typename base_t::reference        reference;
    typedef typename base_t::const_reference  const_reference;
    typedef typename base_t::value_type       value_type;

    template <typename U>
    struct rebind 
    {
      typedef aligned_allocator <U, alignment_> other;
    };

    aligned_allocator () throw ()
    : base_t ()
    {
    }

    aligned_allocator (const aligned_allocator &o) throw ()
    : base_t (o)
    {
    }

    ~aligned_allocator () throw ()
    {
    }

    pointer
    address(reference x) const 
    { 
      return &x; 
    }

    const_pointer
    address(const_reference x) const 
    { 
      return &x; 
    }

    pointer 
    allocate (size_type count, const void * = 0)
    {
      return (pointer)BS_KERNEL.get_memory_manager ().allocate_aligned (sizeof (T) * count, alignment_);
    }

    void 
    deallocate (pointer ptr, size_type)
    {
      BS_KERNEL.get_memory_manager ().deallocate_aligned (ptr);
    }
  };

  template<typename T, size_t alignment_>
  inline bool
  operator== (const aligned_allocator <T, alignment_> &,
              const aligned_allocator <T, alignment_> &)
  { 
    return true; 
  }
  
  template<typename T, size_t alignment_>
  inline bool
  operator!= (const aligned_allocator <T, alignment_> &,
              const aligned_allocator <T, alignment_> &)
  { 
    return false; 
  }

} // namespace blue_sky

#endif  // #ifndef BS_TOOLS_ALIGNED_ALLOCATOR_H_

