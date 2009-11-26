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

namespace blue_sky {

  template <typename T>
  struct BS_API_PLUGIN shared_array_manager
  {
    static shared_array_manager <T> *
    instance ();

    shared_array_manager ();
    ~shared_array_manager ();

    void
    add_array (T *array, size_t size, void *owner);

    bool
    rem_array (T *array, void *owner);

    void
    change_array (T *array, T *new_memory, T *new_finish, const long long &new_capacity);

    void
    change_array_end (T *array, T *new_finish);

    struct impl;
    impl *impl_;
  };

  namespace detail {

    template <typename T, size_t align>
    inline bool
    is_owner__ (const aligned_allocator <T, align> &, const long long &holder)
    {
      return (holder & 1) != 0;
    }

    template <typename T, size_t align>
    inline void
    set_owner__ (const aligned_allocator <T, align> &, bool is_owner, long long &holder)
    {
      if ((holder & 1) != 0)
        {
          holder += 1 + is_owner;
        }
      else
        {
          holder += is_owner;
        }
    }

    template <typename T, size_t align>
    inline void
    add_owner__ (const aligned_allocator <T, align> &, T *ownee, size_t size, void *owner)
    {
      shared_array_manager <T>::instance ()->add_array (ownee, size, owner);
    }

    template <typename T, size_t align>
    inline bool
    rem_owner__ (const aligned_allocator <T, align> &, T *ownee, void *owner)
    {
      return shared_array_manager <T>::instance ()->rem_array (ownee, owner);
    }

    template <typename T, size_t align>
    inline long long
    new_capacity__ (const aligned_allocator <T, align> &a, size_t size, size_t i, bool is_owner)
    {
      long long capacity = size + (std::max) (size, i);
      set_owner__ (a, is_owner, capacity);
      return capacity;
    }

    template <typename T, size_t align>
    inline void
    change_ownee__ (const aligned_allocator <T, align> &, T *ownee, T *new_memory, T *new_finish, const long long &new_capacity)
    {
      shared_array_manager <T>::instance ()->change_array (ownee, new_memory, new_finish, new_capacity);
    }

    template <typename T, size_t align>
    inline void
    change_ownee__ (const aligned_allocator <T, align> &, T *ownee, T *new_finish)
    {
      shared_array_manager <T>::instance ()->change_array_end (ownee, new_finish);
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
    typedef std::size_t     size_type;
    typedef std::ptrdiff_t  difference_type;
    typedef allocator_type  allocator_t;

    struct internal_deleter
    {
      enum {
        owner = true, 
      };

      void
      operator () (array_ext <T> *t)
      {
        allocator_t allocator_;

        detail::destroy (t->begin (), t->end (), allocator_);
        detail::deallocate (t->begin (), t->capacity_, allocator_);

        delete t;
      }
    };

    struct numpy_deleter
    {
      enum {
        owner = false,
      };

      void
      operator () (array_ext <T> *t)
      {
        delete t;
      }
    };

    ~shared_array ()
    {
      if (detail::rem_owner__ (allocator_, array_, this))
        {
          detail::destroy (begin (), end (), allocator_);
          detail::deallocate (begin (), capacity_, allocator_);
        }
    }

    shared_array ()
    : array_ (0)
    , array_end_ (0)
    , capacity_ (0)
    {
      detail::set_owner__ (allocator_, true, capacity_);
      detail::add_owner__ (allocator_, array_, 0, this);
    }

    shared_array (const internal_deleter &d, T *e, size_t N)
    : array_ (e)
    , array_end_ (e + N)
    , capacity_ (N)
    {
      detail::set_owner__ (allocator_, true, capacity_);
      detail::add_owner__ (allocator_, array_, N, this);
    }

    shared_array (const numpy_deleter &d, T *e, size_t N)
    : array_ (e)
    , array_end_ (e + N)
    , capacity_ (N)
    {
      detail::set_owner__ (allocator_, false, capacity_);
    }

    shared_array (const shared_array &v)
    : array_ (v.array_)
    , array_end_ (v.array_end_)
    , capacity_ (v.size ())
    {
      if (detail::is_owner__ (allocator_, v.capacity_))
        {
          detail::set_owner__ (allocator_, true, capacity_);
          detail::add_owner__ (allocator_, array_, v.size (), this);
        }

      BS_ASSERT (capacity_ == v.capacity_);
    }

    shared_array &
    operator= (const shared_array &v)
    {
      if (detail::rem_owner__ (allocator_, array_, this))
        {
          detail::destroy (begin (), end (), allocator_);
          detail::deallocate (begin (), capacity_, allocator_);
        }

      array_      = v.array_;
      array_end_  = v.array_end_;
      capacity_   = v.capacity_;

      if (detail::is_owner__ (allocator_, v.capacity_))
        {
          detail::add_owner__ (allocator_, array_, v.size (), this);
        }

      BS_ASSERT (capacity_ == v.capacity_) (capacity_) (v.capacity_);
      return *this;
    }

    long long
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
    reference operator[](size_type i)
    {
      return array_[i];
    }

    const_reference operator[](size_type i) const
    {
      return array_[i];
    }

    // at() with range check
    reference at(size_type i)
    {
      rangecheck (i);
      return array_[i];
    }
    const_reference at(size_type i) const
    {
      rangecheck (i);
      return array_[i];
    }
    // size is constant
    size_type size() const
    {
      return size_type (array_end_ - array_);
      //return array_->size ();
    }
    bool empty() const
    {
      return array_ == array_end_;
      //return array_->empty ();
    }
    size_type max_size() const
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
    rangecheck (size_type i) const
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
      return detail::is_owner__ (allocator_, capacity_);
    }

    T *
    allocate (size_type count)
    {
      count += allocator_t::alignment_size * 2;
    }


  public:
    T             *array_;
    T             *array_end_;
    long long     capacity_;
    allocator_t   allocator_;
  };

} // namespace blue_sky

#endif // #ifndef BS_TOOLS_SHARED_ARRAY_H_

