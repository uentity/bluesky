/**
 * \file allocator_interface.h
 * \brief interface for backend memory allocators
 * \author Sergey Miryanov
 * \date 18.06.2009
 * */
#ifndef BS_ALLOCATOR_INTERFACE_H_
#define BS_ALLOCATOR_INTERFACE_H_

#include "memory_manager.h"

namespace blue_sky {

  struct memory_manager::allocator_interface
  {
    virtual ~allocator_interface () 
    {
    }

    virtual void *
    allocate (size_t size) = 0;

    virtual void 
    deallocate (void *ptr) = 0;

    bool 
    empty ()
    {
      BS_ASSERT (count_ >= 0) (count_);
      return count_ == 0;
    }

  protected:

    long count_;
  };


} // namespace blue_sky

#endif // #ifndef BS_ALLOCATOR_INTERFACE_H_

