/**
 * \file memory_manager.h
 * \brief 
 * \author Sergey Miryanov
 * \date 27.01.2009
 * */
#ifndef BS_MEMORY_MANAGER_H_
#define BS_MEMORY_MANAGER_H_

#include "bs_common.h"

namespace blue_sky {

  struct BS_API memory_manager 
  {
    struct allocator_info;
    struct allocator_interface;

    typedef size_t                                  thread_id_t;
    typedef std::map <thread_id_t, allocator_info*> allocator_info_map_t;
    typedef size_t                                  ptr_diff_t;

    memory_manager ();

    ~memory_manager ();

    /** 
     * \brief set backend memory allocator
     * \return false if alloc_ != dealloc_ (if previous
     *         operation of set allocator not completed yet)
     * */
    bool 
    set_backend_allocator (allocator_interface *alloc_);

    void *
    allocate (size_t size);

    void *
    allocate_aligned (size_t size, size_t alignment);

    void
    deallocate (void *ptr);

    void
    deallocate_aligned (void *ptr);

    void
    store_allocate_info (void *ptr, size_t size);

    void
    store_deallocate_info (void *ptr);

    allocator_info *
    get_allocator_info (allocator_info_map_t &locked_info, thread_id_t thread_id);

    void
    print_allocate (void *ptr, size_t size);

    void
    print_deallocate (void *ptr);

    void
    print_info (allocator_info *info, bool print_map = true);

    void
    print_info (bool print_map = true);

    //static memory_manager &
    //instance ();

  private:
    allocator_info_map_t    allocator_info_map;
    allocator_interface     *alloc_;
    allocator_interface     *dealloc_;
  };


} // namespace blue_sky

inline void *
operator new (size_t nbytes, blue_sky::memory_manager &mgr) 
{
  return mgr.allocate_aligned (nbytes, 16);
}

inline void
operator delete (void *ptr, blue_sky::memory_manager &mgr) 
{
  mgr.deallocate_aligned (ptr);
}

#endif  // #ifndef BS_MEMORY_MANAGER_H_

