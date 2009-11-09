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

    shared_array (const internal_deleter &d = internal_deleter ())
    : array_ (new array_ext <T> (0, 0), d)
    , owned_ (d.owner)
    {
    }

    shared_array (const internal_deleter &d, T *e, size_t N)
    : array_ (new array_ext <T> (e, N), d)
    , owned_ (d.owner)
    {
    }
    shared_array (const numpy_deleter &d, T *e, size_t N)
    : array_ (new array_ext <T> (e, N), d)
    , owned_ (d.owner)
    {
    }

    shared_array (const shared_array &v)
    : array_ (v.array_)
    , owned_ (v.owned_)
    {
    }
    
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
    shared_array &operator= (const shared_array &rhs)
    {
      array_ = rhs.array_;
      owned_ = rhs.owned_;
      return *this;
    }
    // assign one value to all elements
    void assign (const T& value)
    {
      array_->assign (value);
    }

  public:
    boost::shared_ptr <array_ext <T> >  array_;
    bool                                owned_;
  };

} // namespace blue_sky

#endif // #ifndef BS_TOOLS_SHARED_ARRAY_H_

