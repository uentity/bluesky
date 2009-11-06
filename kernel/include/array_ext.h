/**
 * \file array_ext.h
 * \brief array that holds pointer to data with interface like boost::array
 * \author Sergey Miryanov
 * \date 20.02.2009
 * */

#ifndef BS_TOOLS_ARRAY_EXT_H_
#define BS_TOOLS_ARRAY_EXT_H_

#include "bs_assert.h"

namespace blue_sky {

  template<class T>
  class array_ext
  {
  public:
    // type definitions
    typedef T              value_type;
    typedef T*             iterator;
    typedef const T*       const_iterator;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    // iterator support
    iterator begin()
    {
      return elems;
    }
    const_iterator begin() const
    {
      return elems;
    }
    iterator end()
    {
      return elems+N;
    }
    const_iterator end() const
    {
      return elems+N;
    }

    // reverse iterator support
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    array_ext (T *e = 0, size_t N = 0) 
    : elems (e)
    , N (N)
    , capacity_ (N)
    {
    }

    T *
    init (T *d)
    {
      elems = d;
      return elems + N;
    }

    reverse_iterator rbegin()
    {
      return reverse_iterator(end());
    }
    const_reverse_iterator rbegin() const
    {
      return const_reverse_iterator(end());
    }
    reverse_iterator rend()
    {
      return reverse_iterator(begin());
    }
    const_reverse_iterator rend() const
    {
      return const_reverse_iterator(begin());
    }

    // operator[]
    reference operator[](size_type i)
    {
      BS_ASSERT( i < N && N != 0 && "out of range" );
      return elems[i];
    }

    const_reference operator[](size_type i) const
    {
      BS_ASSERT( i < N && N != 0 && "out of range" );
      return elems[i];
    }

    // at() with range check
    reference at(size_type i)
    {
      rangecheck(i);
      return elems[i];
    }
    const_reference at(size_type i) const
    {
      rangecheck(i);
      return elems[i];
    }

    // front() and back()
    reference front()
    {
      return elems[0];
    }

    const_reference front() const
    {
      return elems[0];
    }

    reference back()
    {
      return elems[N-1];
    }

    const_reference back() const
    {
      return elems[N-1];
    }

    // size is constant
    size_type size() const
    {
      return N;
    }
    bool empty() const
    {
      return N == 0;
    }
    size_type max_size() const
    {
      return N;
    }

    // swap (note: linear complexity)
    void swap (array_ext<T> &y)
    {
      BS_ASSERT (y.size () == size ()) (y.size ()) (size ());
      std::swap_ranges(begin(),end(), y.begin());
    }

    // direct access to data (read-only)
    const T* data() const
    {
      return elems;
    }
    T* data()
    {
      return elems;
    }

    // use array as C array (direct read/write access to data)
    T* c_array()
    {
      return elems;
    }

    //// assignment with type conversion
    //template <typename T2>
    //array_ext<T> &operator= (const array_ext<T2>& rhs)
    //{
    //  BS_ASSERT (rhs.size () == size ()) (rhs.size ()) (size ());
    //  std::copy(rhs.begin(),rhs.end(), begin());
    //  return *this;
    //}

    array_ext <T> &operator= (array_ext <T> &rhs)
    {
      BS_ASSERT (rhs.size () == size ()) (rhs.size ()) (size ());

      elems     = rhs.elems;
      N         = rhs.N;
      capacity_ = rhs.capacity_;

      return *this;
    }

    // assign one value to all elements
    void assign (const T& value)
    {
      std::fill_n(begin(),size(),value);
    }

    // check range (may be private because it is static)
    void rangecheck (size_type i)
    {
      if (i >= size())
        {
          throw std::out_of_range("array<>: index out of range");
        }
    }
  public:
    T             *elems;
    size_t        N;
    size_t        capacity_;
  };

  // global swap()
  template<class T>
  inline void swap (array_ext<T>& x, array_ext<T>& y)
  {
    BS_ASSERT (x.size () == y.size ()) (x.size ()) (y.size ());
    x.swap(y);
  }
}

#endif // BS_TOOLS_ARRAY_EXT_H_

