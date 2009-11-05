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

namespace blue_sky {

  template <typename T>
  struct shared_array 
  {
    // type definitions
    typedef T              value_type;
    typedef T*             iterator;
    typedef const T*       const_iterator;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    struct internal_deleter
    {
      enum {
        owner = true, 
      };

      void
      operator () (array_ext <T> *t);
    };

    struct numpy_deleter
    {
      enum {
        owner = false,
      };

      void
      operator () (array_ext <T> *t);
    };

    shared_array ()
    : array_ (new array_ext <T> (0, 0))
    {
    }

    template <typename D>
    shared_array (const D &d, T *e = 0, size_t N = 0)
    : array_ (new array_ext <T> (e, N)/*, d*/)
    {
    }
    
    //shared_array (T *e = 0, size_t N = 0)
    //: array_ (new array_ext <T> (e, N))
    //{
    //}

    // iterator support
    iterator begin()
    {
      return array_->begin ();
    }
    const_iterator begin() const
    {
      return array_->begin ();
    }
    iterator end()
    {
      return array_->end ();
    }
    const_iterator end() const
    {
      return array_->end ();
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
      return array_->front ();
    }

    const_reference front() const
    {
      return array_->front ();
    }

    reference back()
    {
      return array_->back ();
    }

    const_reference back() const
    {
      return array_->back ();
    }
    
    // operator[]
    reference operator[](size_type i)
    {
      return array_->operator[] (i);
    }

    const_reference operator[](size_type i) const
    {
      return array_->operator[] (i);
    }

    // at() with range check
    reference at(size_type i)
    {
      return array_->at (i);
    }
    const_reference at(size_type i) const
    {
      return array_->at (i);
    }
    // size is constant
    size_type size() const
    {
      return array_->size ();
    }
    bool empty() const
    {
      return array_->empty ();
    }
    size_type max_size() const
    {
      return array_->max_size ();
    }

    // direct access to data (read-only)
    const T* data() const
    {
      return array_->data ();
    }
    T* data()
    {
      return array_->data ();
    }

    // use array as C array (direct read/write access to data)
    T* c_array()
    {
      return array_->c_array ();
    }
    shared_array <T> &operator= (const shared_array <T> &rhs)
    {
      (*array_) = (*rhs.array_);
      return *this;
    }
    // assign one value to all elements
    void assign (const T& value)
    {
      array_->assign (value);
    }

  public:
    boost::shared_ptr <array_ext <T> > array_;
  };


  typedef unsigned char             uint8_t;
  typedef float                     float16_t;

  typedef shared_array <uint8_t>    array_uint8_t;
  typedef shared_array <float16_t>  array_float16_t;

} // namespace blue_sky

#include "shared_array_allocator.h"


#endif // #ifndef BS_TOOLS_SHARED_ARRAY_H_

