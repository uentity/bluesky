/**
 * */
#ifndef BS_SHARED_ARRAY_MANAGER_H_
#define BS_SHARED_ARRAY_MANAGER_H_

#include "bs_kernel_tools.h"

namespace blue_sky {

  struct shared_array_manager::impl
  {
    void
    add_array (void *array, size_t size, const owner_t &owner)
    {
      //if ((size & 1) == 0)
      //  {
      //    std::cout << "Is not owner" << std::endl;
      //  }

      for (size_t i = 0, cnt = arrays_.size (); i < cnt; ++i)
        {
          array_info &info = arrays_[i];
          if (info.array_ == array)
            {
              info.owners_.push_back (owner);
              //std::cout << "Add new owner " << owner.array << " to " << array << std::endl;
              //std::cout << kernel_tools::get_backtrace (32) << std::endl;

              return ;
            }
        }

      array_info info;
      info.array_ = array;
      info.size_ = size;
      info.owners_.push_back (owner);
      arrays_.push_back (info);

      //std::cout << "Add owner " << owner.array << " to array " << array << std::endl;
      //std::cout << kernel_tools::get_backtrace (32) << std::endl;
    }

    bool
    rem_array (void *array, void *owner)
    {
      //std::cout << "Try to remove owner " << owner << " from array " << array << std::endl;
      for (size_t i = 0, cnt = arrays_.size (); i < cnt; ++i)
        {
          array_info &info = arrays_[i];
          if (info.array_ == array)
            {
              rem_owner (info.owners_, owner);
              //std::cout << "Remove owner " << owner << " from array " << array << std::endl;

              if (info.owners_.empty ())
                {
                  //std::cout << "Deallocate memory " << array << std::endl;
                  arrays_.erase (arrays_.begin () + i);
                  return true;
                }
            }
        }

      return false;
    }

    void
    rem_owner (std::vector <owner_t> &owners, void *owner)
    {
      for (size_t i = 0, cnt = owners.size (); i < cnt; ++i)
        {
          if (owners[i].array == owner)
            {
              owners.erase (owners.begin () + i);
              break;
            }
        }
    }

    void
    change_array (void *array, void *new_memory, void *new_finish, const size_t &new_capacity)
    {
      for (size_t i = 0, cnt = arrays_.size (); i < cnt; ++i)
        {
          array_info &info = arrays_[i];
          if (info.array_ == array)
            {
              info.size_ = new_capacity;
              info.array_ = new_memory;
              for (size_t j = 0, jcnt = info.owners_.size (); j < jcnt; ++j)
                {
                  owner_t &owner = info.owners_[j];

                  *(reinterpret_cast <char **> (owner.array))     = reinterpret_cast <char *> (new_memory);
                  *(reinterpret_cast <char **> (owner.array_end)) = reinterpret_cast <char *> (new_finish);
                  *(reinterpret_cast <char **> (owner.capacity))  = reinterpret_cast <char *> (new_capacity);
                }

              break;
            }
        }
    }

    void
    change_array_end (void *array, void *new_finish)
    {
      for (size_t i = 0, cnt = arrays_.size (); i < cnt; ++i)
        {
          array_info &info = arrays_[i];
          if (info.array_ == array)
            {
              for (size_t j = 0, jcnt = info.owners_.size (); j < jcnt; ++j)
                {
                  *(reinterpret_cast <char **> (info.owners_[j].array_end)) = reinterpret_cast <char *> (new_finish);
                }

              break;
            }
        }
    }

    void
    print ()
    {
      for (size_t i = 0, cnt = arrays_.size (); i < cnt; ++i)
        {
          const array_info &info = arrays_[i];
          std::cout << "For array " << info.array_ << " following owners registered: " << std::endl;
          for (size_t j = 0, jcnt = info.owners_.size (); j < jcnt; ++j)
            {
              std::cout << "\t" << info.owners_[j].array << std::endl;
            }
        }
    }

    struct array_info
    {
      void                   *array_;
      size_t                 size_;
      std::vector <owner_t>  owners_;
    };

    std::vector <array_info> arrays_;
  };

} // namespace blue_sky



#endif // #ifndef BS_SHARED_ARRAY_MANAGER_H_

