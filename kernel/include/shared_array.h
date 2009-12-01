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

  namespace detail {

    inline size_t
    new_capacity__ (size_t size, size_t i, bool is_owner)
    {
      return size + (std::max) (size, i);
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
      if (owner_list_ && remove_owner ())
        {
          detail::detail_t <boost::is_arithmetic <T>::value>::destroy (begin (), end (), allocator_);
          detail::deallocate (begin (), capacity_, allocator_);

          delete owner_list_;
        }
    }

    shared_array ()
    : array_ (0)
    , array_end_ (0)
    , capacity_ (0)
    , owner_list_ (new owner_list_t)
    {
      BS_ASSERT (owner_list_);
      owner_list_->push_back (this);
    }

    shared_array (const numpy_deleter &d, T *e, size_t N)
    : array_ (e)
    , array_end_ (e + N)
    , capacity_ (N)
    , owner_list_ (0)
    {
    }

    shared_array (const shared_array &v)
    : array_ (v.array_)
    , array_end_ (v.array_end_)
    , capacity_ (v.capacity_)
    , owner_list_ (v.owner_list_)
    {
      if (owner_list_)
        {
          owner_list_->push_back (this);
        }
    }

    shared_array &
    operator= (const shared_array &v)
    {
      if (owner_list_ && remove_owner ())
        {
          detail::detail_t <boost::is_arithmetic <T>::value>::destroy (begin (), end (), allocator_);
          detail::deallocate (begin (), capacity_, allocator_);

          delete owner_list_;
        }

      array_      = v.array_;
      array_end_  = v.array_end_;
      capacity_   = v.capacity_;
      owner_list_ = v.owner_list_;

      if (owner_list_)
        {
          owner_list_->push_back (this);
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
      return owner_list_;
    }

    bool
    remove_owner ()
    {
      for (size_t i = 0, cnt = owner_list_->size (); i < cnt; ++i)
        {
          if (owner_list_->operator[] (i) == this)
            {
              owner_list_->erase (owner_list_->begin () + i);
              break;
            }
        }

      return owner_list_->empty ();
    }

    void
    change_owner (T *new_memory, T *new_finish, const size_t &new_capacity)
    {
      owner_list_t &list = *owner_list_;
      for (size_t i = 0, cnt = list.size (); i < cnt; ++i)
        {
          owner_t owner     = list[i];
          owner->array_     = new_memory;
          owner->array_end_ = new_finish;
          owner->capacity_  = new_capacity;
        }
    }

    void
    change_owner (T *new_finish)
    {
      owner_list_t &list = *owner_list_;
      for (size_t i = 0, cnt = list.size (); i < cnt; ++i)
        {
          list[i]->array_end_ = new_finish;
        }
    }

    typedef shared_array <T> *    owner_t;
    typedef std::vector <owner_t> owner_list_t;

  public:
    T             *array_;
    T             *array_end_;
    size_t        capacity_;
    owner_list_t  *owner_list_;
    allocator_t   allocator_;
  };

} // namespace blue_sky

#endif // #ifndef BS_TOOLS_SHARED_ARRAY_H_

