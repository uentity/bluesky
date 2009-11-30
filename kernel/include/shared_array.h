/**
 * \file shared_array.h
 * \brief shared array
 * \author Sergey Miryanov
 * \date 28.05.2009
 * */
#ifndef BS_TOOLS_SHARED_ARRAY_H_
#define BS_TOOLS_SHARED_ARRAY_H_

#include <boost/shared_ptr.hpp>
#include "array_ext.h"
#include "shared_array_detail.h"
#include "aligned_allocator.h"

#include <boost/type_traits.hpp>

namespace blue_sky {

  struct BS_API_PLUGIN shared_array_manager
  {
    static shared_array_manager *
    instance ();

    shared_array_manager ();
    ~shared_array_manager ();

    struct owner_t
    {
      owner_t (void *array, void *array_end, size_t *capacity)
      : array (array)
      , array_end (array_end)
      , capacity (capacity)
      {
      }

      void    *array;
      void    *array_end;
      size_t  *capacity;
    };

    void
    add_array (void *array, size_t size, const owner_t &owner);

    bool
    rem_array (void *array, void *owner);

    void
    change_array (void *array, void *new_memory, void *new_finish, const size_t &new_capacity);

    void
    change_array_end (void *array, void *new_finish);

    struct impl;
    impl *impl_;
  };

  namespace detail {

    inline bool
    is_owner__ (const size_t &holder)
    {
      return (holder & 1) != 0;
    }

    inline size_t
    owner__ (bool is_owner, const size_t &holder)
    {
      if ((holder & 1) != 0)
        {
          return holder + 1 + is_owner;
        }
      else
        {
          return holder + is_owner;
        }
    }

    template <typename T>
    inline void
    add_owner__ (T *ownee, size_t size, T **owner, T **array_end, size_t *capacity)
    {
      shared_array_manager::instance ()->add_array (ownee, size, shared_array_manager::owner_t (owner, array_end, capacity));
    }

    template <typename T>
    inline bool
    rem_owner__ (T *ownee, T **owner)
    {
      return shared_array_manager::instance ()->rem_array (ownee, owner);
    }

    inline size_t
    new_capacity__ (size_t size, size_t i, bool is_owner)
    {
      return owner__ (is_owner, size + (std::max) (size, i));
    }

    template <typename T>
    inline void
    change_owner__ (T *ownee, T *new_memory, T *new_finish, const size_t &new_capacity)
    {
      shared_array_manager::instance ()->change_array (ownee, new_memory, new_finish, new_capacity);
    }

    template <typename T>
    inline void
    change_owner__ (T *ownee, T *new_finish)
    {
      shared_array_manager::instance ()->change_array_end (ownee, new_finish);
    }
  }


  template <typename T, typename allocator_type = aligned_allocator <T, 16> >
  struct shared_array 
  {
    // type definitions
    typedef T               value_type;
    typedef T*              iterator;
    typedef const T*        const_iterator;
    typedef T&              reference;
    typedef const T&        const_reference;
    typedef std::ptrdiff_t  difference_type;
    typedef allocator_type  allocator_t;
    typedef size_t          size_type;

    struct numpy_deleter
    {
    };

    ~shared_array ()
    {
      if (detail::rem_owner__ (array_, &array_))
        {
          detail::detail_t <boost::is_arithmetic <T>::value>::destroy (begin (), end (), allocator_);
          detail::deallocate (begin (), capacity_, allocator_);
        }
    }

    shared_array ()
    : array_ (0)
    , array_end_ (0)
    , capacity_ (detail::owner__ (true, 0))
    {
      detail::add_owner__ (array_, 0, &array_, &array_end_, &capacity_);
    }

    shared_array (const numpy_deleter &d, T *e, size_t N)
    : array_ (e)
    , array_end_ (e + N)
    , capacity_ (detail::owner__ (false, N))
    {
    }

    shared_array (const shared_array &v)
    : array_ (v.array_)
    , array_end_ (v.array_end_)
    , capacity_ (v.capacity_)
    {
      if (detail::is_owner__ (v.capacity_))
        {
          detail::add_owner__ (array_, v.capacity (), &array_, &array_end_, &capacity_);
        }
    }

    shared_array &
    operator= (const shared_array &v)
    {
      if (detail::is_owner__ (capacity_) && detail::rem_owner__ (array_, &array_))
        {
          detail::detail_t <boost::is_arithmetic <T>::value>::destroy (begin (), end (), allocator_);
          detail::deallocate (begin (), capacity_, allocator_);
        }

      array_      = v.array_;
      array_end_  = v.array_end_;
      capacity_   = v.capacity_;

      if (detail::is_owner__ (v.capacity_))
        {
          detail::add_owner__ (array_, v.capacity (), &array_, &array_end_, &capacity_);
        }

      return *this;
    }

    size_t
    capacity () const
    {
      return capacity_;
    }

    // iterator support
    iterator begin()
    {
      return iterator (array_);
    }
    const_iterator begin() const
    {
      return const_iterator (array_);
    }
    iterator end()
    {
      return iterator (array_end_);
    }
    const_iterator end() const
    {
      return const_iterator (array_end_);
    }

    //reverse_iterator rbegin()
    //{
    //  return reverse_iterator(end());
    //}
    //const_reverse_iterator rbegin() const
    //{
    //  return const_reverse_iterator(end());
    //}
    //reverse_iterator rend()
    //{
    //  return reverse_iterator(begin());
    //}
    //const_reverse_iterator rend() const
    //{
    //  return const_reverse_iterator(begin());
    //}

    // front() and back()
    reference front()
    {
      return *begin ();
    }

    const_reference front() const
    {
      return *begin ();
    }

    reference back()
    {
      return *(end () - 1);
    }

    const_reference back() const
    {
      return *(end () - 1);
    }
    
    // operator[]
    reference operator[](size_t i)
    {
      return array_[i];
    }

    const_reference operator[](size_t i) const
    {
      return array_[i];
    }

    // at() with range check
    reference at(size_t i)
    {
      rangecheck (i);
      return array_[i];
    }
    const_reference at(size_t i) const
    {
      rangecheck (i);
      return array_[i];
    }
    // size is constant
    size_t size() const
    {
      return size_t (array_end_ - array_);
      //return array_->size ();
    }
    bool empty() const
    {
      return array_ == array_end_;
      //return array_->empty ();
    }
    size_t max_size() const
    {
      return size ();
      //return array_->max_size ();
    }

    // direct access to data (read-only)
    const T* data() const
    {
      return array_;
      //return array_->data ();
    }
    T* data()
    {
      return array_;
      //return array_->data ();
    }

    // use array as C array (direct read/write access to data)
    T* c_array()
    {
      return array_;
      //return array_->c_array ();
    }

    // assign one value to all elements
    void assign (const T& value)
    {
      //array_->assign (value);
      std::fill_n (begin (), size (), value);
    }

  private:
    // check range 
    void 
    rangecheck (size_t i) const
    {
      if (i >= size())
        {
          throw std::out_of_range("shared_array: index out of range");
        }
    }

  protected:
    bool 
    is_owner () const
    {
      return detail::is_owner__ (capacity_);
    }


  public:
    T             *array_;
    T             *array_end_;
    size_t        capacity_;
    allocator_t   allocator_;
  };

} // namespace blue_sky

#include "shared_array_manager.h"

#endif // #ifndef BS_TOOLS_SHARED_ARRAY_H_

