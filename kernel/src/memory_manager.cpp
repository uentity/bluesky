/** 
 * \file memory_manager.cpp
 * \brief impl of
 * \author Sergey Miryanov
 * \date 10.06.2009
 * */
#ifdef BSPY_EXPORTING_PLUGIN
#include <boost/python.hpp>
#endif

#include "memory_manager.h"
#include "bs_exception.h"
#include "bs_assert.h"
#include "bs_report.h"
#include "bs_kernel.h"

// to avoid DUMB linker error we have to include bs_object_base.h
#include "bs_object_base.h"
#include "bs_tree.h"

#include "allocator_interface.h"

#if defined (BS_BOS_CORE_TLSF_ALLOCATOR)
#include "tlsf_allocator.h"
#elif defined (BS_BOS_CORE_HOARD_ALLOCATOR)
#include "hoard_allocator.h"
#elif defined (BS_BOS_CORE_DLMALLOC_ALLOCATOR)
#include "dlmalloc_allocator.h"
#elif defined (BS_BOS_CORE_TCMALLOC_ALLOCATOR)
#include "tcmalloc_allocator.h"
#else
#include "std_allocator.h"
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef BS_BOS_CORE_COLLECT_BACKTRACE
#ifdef _WIN32
#include "backtrace_tools_win.h"
#else
#include "backtrace_tools_unix.h"
#endif
#endif

#include "get_thread_id.h"

namespace blue_sky {

#ifndef BS_BACKTRACE_LEN
#define BS_BACKTRACE_LEN 14
#endif

  struct memory_manager::allocator_info
  {
#ifdef BS_BOS_CORE_COLLECT_BACKTRACE
    enum { max_backtrace_len = BS_BACKTRACE_LEN, };
    struct alloc_info
    {
      size_t    size_;
      void      *backtrace_[max_backtrace_len];
      size_t    backtrace_len_;
    };
#else
    struct alloc_info
    {
      size_t    size_;
    };
#endif

    struct dealloc_info
    {
      size_t alloc_count;
      size_t dealloc_count;

      dealloc_info ()
      : alloc_count (0),
      dealloc_count (0)
      {

      }
    };

    typedef std::map <void *, alloc_info>   alloc_map_t;
    typedef std::map <void *, dealloc_info> dealloc_map_t;

    allocator_info ()
      : alloc_call_count (0),
      dealloc_call_count (0),
      total_alloc_size (0),
      total_dealloc_size (0),
      alloc_size (0),
      max_alloc_size (0)
    {
    }

    size_t          alloc_call_count;
    size_t          dealloc_call_count;
    size_t          total_alloc_size;
    size_t          total_dealloc_size;
    size_t          alloc_size;
    size_t          max_alloc_size;
    alloc_map_t     alloc_map;
    dealloc_map_t   dealloc_map;
  };

  memory_manager::memory_manager ()
  : alloc_ (0),
  dealloc_ (0)
  {
    alloc_    = &allocator_t::instance ();
    dealloc_  = &allocator_t::instance ();
  }

  memory_manager::~memory_manager ()
  {
  }

  bool 
  memory_manager::set_backend_allocator (allocator_interface *a)
  {
    if (alloc_ != dealloc_)
      {
        return false;
      }

#ifdef BS_BOS_CORE_DEBUG_MEMORY
    BSOUT << "memory_manager: swap memory allocators begin" << bs_end;
#endif

    dealloc_  = alloc_;
    alloc_    = a;
    return true;
  }

  void *
  memory_manager::allocate (size_t size)
  {
    char *ptr = (char *)alloc_->allocate (size);
    if (!ptr)
      {
#ifdef BS_BOS_CORE_DEBUG_MEMORY
        print_info ();
#endif
        throw bs_exception ("memory_manager::allocate_aligned", "Can't allocate memory");
      }

#ifdef BS_BOS_CORE_DEBUG_MEMORY
    store_allocate_info (ptr, size);
#endif

    return ptr;
  }

  void
  memory_manager::deallocate (void *ptr)
  {
    if (!ptr)
      {
        BS_ASSERT (ptr);
        return ;
      }

    dealloc_->deallocate (ptr);

    if (alloc_ != dealloc_ && dealloc_->empty ())
      {
#ifdef BS_BOS_CORE_DEBUG_MEMORY
        BSOUT << "memory_manager: swap memory allocators complete" << bs_end;
#endif
        dealloc_ = alloc_;
      }

#ifdef BS_BOS_CORE_DEBUG_MEMORY
    store_deallocate_info (ptr);
#endif
  }

  void *
  memory_manager::allocate_aligned (size_t size, size_t alignment)
  {
    size += alignment - 1;
    size += sizeof (ptr_diff_t);

    char *ptr = (char *)allocate (size);
    BS_ASSERT (ptr);

    ptr_diff_t new_ptr = (ptr_diff_t)ptr + sizeof (ptr_diff_t);
    char *aligned_ptr = (char *)((new_ptr + alignment - 1) & ~(alignment - 1));

    ptr_diff_t diff = aligned_ptr - ptr;
    *((ptr_diff_t *)aligned_ptr - 1) = diff;

    return aligned_ptr;
  }

  void
  memory_manager::deallocate_aligned (void *ptr_)
  {
    if (ptr_ == 0)
      {
        BS_ASSERT (ptr_);
        return ;
      }

    ptr_diff_t diff = *((ptr_diff_t *)ptr_ - 1);
    char *ptr = (char *)ptr_ - diff;

    deallocate (ptr);
  }

  void
  memory_manager::store_allocate_info (void *ptr, size_t size)
  {
    thread_id_t thread_id = detail::get_thread_id ();

    allocator_info *info = get_allocator_info (allocator_info_map, thread_id);
    BS_ASSERT (info) ((size_t)ptr) (size) (thread_id);

    info->alloc_call_count++;
    info->alloc_size += size;
    info->total_alloc_size += size;

    if (info->alloc_size > info->max_alloc_size)
      {
        info->max_alloc_size = info->alloc_size;
      }

    allocator_info::alloc_map_t::iterator address_it = info->alloc_map.find (ptr);
    if (address_it != info->alloc_map.end ())
      {
        print_allocate (ptr, size);
        print_info (info);
        throw bs_exception ("memory_manager::allocate", "allocate aready allocated memory");
      }

    allocator_info::alloc_info ai;
    ai.size_ = size;

#ifdef BS_BOS_CORE_COLLECT_BACKTRACE
    ai.backtrace_len_ = tools::get_backtrace (ai.backtrace_, allocator_info::max_backtrace_len);
#endif

    info->alloc_map.insert (std::make_pair (ptr, ai));

#ifdef BS_BOS_CORE_PRINT_ALLOC_INFO
    BSOUT << "mem_mgr: allocate [" << ptr << " - " << size << "]" << bs_end;
#endif

#ifdef BS_BOS_CORE_STORE_DEALLOCATE_INFO
    allocator_info::dealloc_map_t::iterator dealloc_it = info->dealloc_map.find (ptr);
    if (dealloc_it != info->dealloc_map.end ())
      {
        dealloc_it->second.alloc_count++;
      }
#endif
  }

  void
  memory_manager::store_deallocate_info (void *ptr)
  {
    thread_id_t thread_id = detail::get_thread_id ();

    allocator_info *info = get_allocator_info (allocator_info_map, thread_id);
    BS_ASSERT (info) ((size_t)ptr) (thread_id);

    allocator_info::alloc_map_t::iterator address_it = info->alloc_map.find (ptr);
    if (address_it == info->alloc_map.end ())
      {
        print_deallocate (ptr);
        print_info (info);
        throw bs_exception ("memory_manager::deallocate", "deallocate unknown memory");
      }

#ifdef BS_BOS_CORE_PRINT_ALLOC_INFO
    BSOUT << "mem_mgr: deallocate [" << ptr << " - " << address_it->second.size_ << "]" << bs_end;
#endif

    info->dealloc_call_count++;
    info->total_dealloc_size += address_it->second.size_;
    info->alloc_size -= address_it->second.size_;
    info->alloc_map.erase (address_it);

#ifdef BS_BOS_CORE_STORE_DEALLOCATE_INFO
    allocator_info::dealloc_map_t::iterator dealloc_it = info->dealloc_map.find (ptr);
    if (dealloc_it == info->dealloc_map.end ())
      {
        allocator_info::dealloc_info di;
        di.dealloc_count = 1;
        di.alloc_count = 1;

        info->dealloc_map.insert (std::make_pair (ptr, di));
      }
    else
      {
        dealloc_it->second.dealloc_count++;
      }
#endif
  }

  memory_manager::allocator_info *
  memory_manager::get_allocator_info (allocator_info_map_t &locked_info, thread_id_t thread_id)
  {
    allocator_info *info = 0;
    allocator_info_map_t::iterator it = locked_info.find (thread_id);
    if (it == locked_info.end ())
      {
        info = new allocator_info;
        locked_info.insert (std::make_pair (thread_id, info));
      }
    else
      {
        info = it->second;
      }

    return info;
  }

  void
  memory_manager::print_allocate (void *ptr, size_t size)
  {
    BSOUT << "allocate: " << ptr << " - " << size << bs_end;
  }

  void
  memory_manager::print_deallocate (void *ptr)
  {
    BSOUT << "deallocate: " << ptr << bs_end;
  }

  void
  memory_manager::print_info (allocator_info *info, bool print_map)
  {
#ifdef BS_BOS_CORE_DEBUG_MEMORY
    BSOUT << "allocate_count: " << info->alloc_call_count << bs_end;
    BSOUT << "deallocate_count: " << info->dealloc_call_count << bs_end;
    BSOUT << "total_alloc_size: " << info->total_alloc_size << bs_end;
    BSOUT << "total_dealloc_size: " << info->total_dealloc_size << bs_end;
    BSOUT << "alloc_size: " << info->alloc_size << bs_end;
    BSOUT << "max_alloc_size: " << info->max_alloc_size << bs_end;
    if (print_map)
      {
        BSOUT << "alloc_map: " << bs_end;

        allocator_info::alloc_map_t::iterator it = info->alloc_map.begin (), e = info->alloc_map.end ();
        for (; it != e; ++it)
          {
            const allocator_info::alloc_info &ai = it->second;
            BSOUT  << "\t[" << (void *)it->first << " : " << ai.size_ << "]" << bs_end;

#ifdef BS_BOS_CORE_COLLECT_BACKTRACE
            char **backtrace_names = tools::get_backtrace_names (ai.backtrace_, ai.backtrace_len_);
            for (size_t i = 0; i < ai.backtrace_len_; ++i)
              {
                if (backtrace_names[i] && strlen (backtrace_names[i]))
                  {
                    BSOUT << "\t\t" << i << ": " << backtrace_names[i] << bs_end;
                  }
              }
            free (backtrace_names);
#endif
          }
      }
#endif
  }

  void
  memory_manager::print_info (bool print_map)
  {
#ifdef BS_BOS_CORE_DEBUG_MEMORY
    thread_id_t thread_id = detail::get_thread_id ();

    allocator_info *info = get_allocator_info (allocator_info_map, thread_id);
    BS_ASSERT (info) (thread_id);

    print_info (info, print_map);
#endif
  }

} // namespace blue_sky

