/**
 * \file shared_vector.h
 * \brief vector that can be shared between different objects with like std::vector facilities
 * \author Sergey Miryanov
 * \date 03.11.2009
 * */
#ifndef BS_SHARED_VECTOR_H_
#define BS_SHARED_VECTOR_H_

#include "shared_array.h"

namespace blue_sky {

  template <typename T/*, typename allocator_t*/>
  struct shared_vector : shared_array <T>
  {
    typedef shared_array <T> base_t;
    typedef typename base_t::value_type value_type;
    typedef typename base_t::size_type  size_type;
    typedef typename base_t::iterator   iterator;
    typedef typename std::allocator <T>::pointer    pointer;

    template <bool b>
    struct is_integral__ 
    {
      enum {
        value = b,
      };
    };

    void
    ctor_fill__ (size_type n, const value_type &value)
    {
      T *new_memory = new T [n];

      std::fill_n (new_memory, n, value);
      this->array_->elems = new_memory;

      this->array_->N = n;
      capacity_ = n;
    }

    template <typename input_iterator>
    void
    ctor_range__ (input_iterator first, input_iterator last, std::input_iterator_tag);

    template <typename forward_iterator>
    void
    ctor_range__ (forward_iterator first, forward_iterator last, std::forward_iterator_tag)
    {
      const size_type n = std::distance (first, last);
      T *new_memory = new T [n];
      std::copy (first, last, new_memory);
      this->array_->elems = new_memory;
      this->array_->N = n;
      capacity_ = n;
    }

    template <typename integer_t>
    void
    ctor_dispatch__ (integer_t n, integer_t value, is_integral__ <true>)
    {
      ctor_fill__ (n, value);
    }

    template <typename input_iterator>
    void
    ctor_dispatch__ (input_iterator first, input_iterator last, is_integral__ <false>)
    {
      typedef typename std::iterator_traits <input_iterator>::iterator_category iterator_category_t;
      ctor_range__ (first, last, iterator_category_t ());
    }

    void
    assign_fill__ (size_type n, const value_type &value)
    {
      if (n > capacity_)
        {
          shared_vector tmp (n, value);
          tmp.swap (*this);
        }
      else if (n > this->size ())
        {
          std::fill (this->begin (), this->end (), value);
          std::fill_n (this->end (), n - this->size (), value);
          this->array_->N += (n - this->size ());
        }
      else
        {
          std::fill_n (this->begin (), n, value);
          this->array_->N = n;
        }
    }

    template <typename integer_t>
    void
    assign_dispatch__ (integer_t n, integer_t value, is_integral__ <true>)
    {
      assign_fill__ (n, value);
    }

    template <typename input_iterator>
    void
    assign_range__ (input_iterator first, input_iterator last, std::input_iterator_tag);

    template <typename forward_iterator>
    void
    assign_range__ (forward_iterator first, forward_iterator last, std::forward_iterator_tag)
    {
      const size_type n = std::distance (first, last);
      pointer old_memory = this->array_->elems;
      if (n > capacity_)
        {
          pointer new_memory = new T [n];
          std::copy (first, last, new_memory);
          this->array_->elems = new_memory;
          this->array_->N = n;
          delete [] old_memory;
        }
      else if (this->size () >= n)
        {
          std::copy (first, last, old_memory);
          this->array_->N = n;
        }
      else
        {
          forward_iterator middle = first;
          std::advance (middle, this->size ());
          std::copy (first, last, old_memory);
          pointer new_finish = std::copy (middle, last, this->end ());
          this->array_->N = size_type (new_finish - this->begin ());
        }
    }

    template <typename input_iterator>
    void
    assign_dispatch__ (input_iterator first, input_iterator last, is_integral__ <false>)
    {
      typedef typename std::iterator_traits <input_iterator>::iterator_category iterator_category_t;
      assign_range__ (first, last, iterator_category_t ());
    }

    void
    push_back_value__ (const value_type &value)
    {
      this->array_->elems[this->array_->N] = value;
      ++this->array_->N;
    }

    bool
    valid_size__ ()
    {
      return this->array_->N != capacity_;
    }

    void
    insert_fill__ (iterator pos, size_type n, const value_type &value)
    {
      if (n != 0)
        {
          if ((capacity_ - this->array_->N) >= n)
            {
              size_type elems_after = this->end () - pos;
              pointer old_finish = this->array_->elems + this->array_->N;

              if (elems_after > n)
                {
                  std::copy (old_finish - n, old_finish, old_finish);
                  std::copy_backward (pos, old_finish - n, old_finish);
                  std::fill (pos, pos + n, value);

                  this->array_->N += n;
                }
              else
                {
                  std::fill_n (old_finish, n - elems_after, value);
                  this->array_->N += n - elems_after;
                  std::copy (pos, old_finish, this->end ());
                  this->array_->N += elems_after;
                  std::fill (pos, old_finish, value);
                }
            }
          else
            {
              capacity_ = new_capacity (n);
              T *new_memory = new T[capacity_];

              T *new_finish = std::copy (this->begin (), pos, new_memory);
              std::fill_n (new_finish, n, value);
              new_finish += n;
              new_finish = std::copy (pos, this->end (), new_finish);

              T *old_memory = this->array_->elems;
              this->array_->elems = new_memory;
              this->array_->N += n;

              delete [] old_memory;
            }
        }
    }

    template <typename input_iterator>
    void
    insert_range__ (iterator pos, input_iterator first, input_iterator last, std::input_iterator_tag);

    template <typename forward_iterator>
    void
    insert_range__ (iterator pos, forward_iterator first, forward_iterator last, std::forward_iterator_tag)
    {
      if (first != last)
        {
          const size_type n = std::distance (first, last);
          if ((capacity_ - this->array_->N) >= n)
            {
              const size_type elems_after = this->end () - pos;
              pointer old_finish = this->array_->elems + this->array_->N;
              if (elems_after > n)
                {
                  std::copy (old_finish - n, old_finish, old_finish);
                  std::copy_backward (pos, old_finish - n, old_finish);
                  std::copy (first, last, pos);

                  this->array_->N += n;
                }
              else
                {
                  forward_iterator middle = first;
                  std::advance (middle, elems_after);
                  std::copy (middle, last, this->end ());
                  this->array_->N += n - elems_after;
                  std::copy (pos, old_finish, this->end ());
                  this->array_->N += elems_after;
                  std::copy (first, middle, pos);
                  
                }
            }
          else
            {
              capacity_ = new_capacity (n);
              T *new_memory = new T[capacity_];
              T *new_finish = std::copy (this->begin (), pos, new_memory);
              new_finish = std::copy (first, last, new_finish);
              new_finish = std::copy (pos, this->end (), new_finish);

              T *old_memory = this->array_->elems;
              this->array_->elems = new_memory;
              this->array_->N += n;

              delete [] old_memory;
            }
        }
    }

    template <typename integer_t>
    void
    insert_dispatch__ (iterator position, integer_t n, integer_t value, is_integral__ <true>)
    {
      insert_fill__ (position, n, value);
    }
    template <typename input_iterator>
    void
    insert_dispatch__ (iterator position, input_iterator first, input_iterator last, is_integral__ <false>)
    {
      typedef typename std::iterator_traits <input_iterator>::iterator_category iterator_category_t;
      insert_range__ (position, first, last, iterator_category_t ());
    }

    void
    erase_at_end__ (size_type n)
    {
      this->array_->N -= n;
    }

    iterator
    erase__ (iterator position)
    {
      if (position + 1 != this->end ())
        {
          std::copy (position + 1, this->end (), position);
        }

      --this->array_->N;
      return position;
    }

    iterator
    erase__ (iterator first, iterator last)
    {
      if (last != this->end ())
        {
          std::copy (last, this->end (), first);
        }

      erase_at_end__ (last - first);
      return first;
    }

    void
    resize__ (size_type new_size, const value_type &value)
    {
      if (new_size < this->size())
        erase_at_end__ (this->size () - new_size);
      else
        insert_fill__ (this->end(), new_size - this->size(), value);
    }

    /**
     *  @brief  Add data to the end of the %vector.
     *  @param  x  Data to be added.
     *
     *  This is a typical stack operation.  The function creates an
     *  element at the end of the %vector and assigns the given data
     *  to it.  Due to the nature of a %vector this operation can be
     *  done in constant time if the %vector has preallocated space
     *  available.
     */
    void
    push_back (const value_type &value)
    {
      if (this->array_->N == capacity_)
        {
          capacity_ = new_capacity (1);
          T *new_memory = new T [capacity_];
          T *old_memory = this->array_->elems;

          std::copy (old_memory, old_memory + this->array_->N, new_memory);

          this->array_->elems = new_memory;
          delete [] old_memory;

          //capacity_ = shared_array_allocator <T>::realloc (this);
        }

      push_back_value__ (value);
    }

    /**
     *  @brief  Inserts given value into %vector before specified iterator.
     *  @param  position  An iterator into the %vector.
     *  @param  x  Data to be inserted.
     *  @return  An iterator that points to the inserted data.
     *
     *  This function will insert a copy of the given value before
     *  the specified location.  Note that this kind of operation
     *  could be expensive for a %vector and if it is frequently
     *  used the user should consider using std::list.
     */
    iterator
    insert (iterator pos, const value_type &value)
    {
      size_type n = pos - this->begin ();
      if (pos == this->end () && valid_size__ ())
        {
          push_back_value__ (value);
        }
      else
        {
          //insert__ (pos, value);
          insert_fill__ (pos, size_type (1), value);
        }

      return iterator (this->begin () + n);
    }

    /**
     *  @brief  Inserts a number of copies of given data into the %vector.
     *  @param  position  An iterator into the %vector.
     *  @param  n  Number of elements to be inserted.
     *  @param  x  Data to be inserted.
     *
     *  This function will insert a specified number of copies of
     *  the given data before the location specified by @a position.
     *
     *  Note that this kind of operation could be expensive for a
     *  %vector and if it is frequently used the user should
     *  consider using std::list.
     */
    void
    insert (iterator pos, size_type n, const value_type &value)
    {
      insert_fill__ (pos, n, value);
    }

    /**
     *  @brief  Inserts a range into the %vector.
     *  @param  position  An iterator into the %vector.
     *  @param  first  An input iterator.
     *  @param  last   An input iterator.
     *
     *  This function will insert copies of the data in the range
     *  [first,last) into the %vector before the location specified
     *  by @a pos.
     *
     *  Note that this kind of operation could be expensive for a
     *  %vector and if it is frequently used the user should
     *  consider using std::list.
     */
    template <typename input_iterator>
    void
    insert (iterator position, input_iterator first, input_iterator last)
    {
      typedef is_integral__ <std::numeric_limits <input_iterator>::is_integer> integral_t;
      insert_dispatch__ (position, first, last, integral_t ());
    }
    
    /**
     *  @brief  Remove element at given position.
     *  @param  position  Iterator pointing to element to be erased.
     *  @return  An iterator pointing to the next element (or end()).
     *
     *  This function will erase the element at the given position and thus
     *  shorten the %vector by one.
     *
     *  Note This operation could be expensive and if it is
     *  frequently used the user should consider using std::list.
     *  The user is also cautioned that this function only erases
     *  the element, and that if the element is itself a pointer,
     *  the pointed-to memory is not touched in any way.  Managing
     *  the pointer is the user's responsibility.
     */
    iterator
    erase (iterator position)
    {
      erase__ (position);
    }

    /**
     *  @brief  Remove a range of elements.
     *  @param  first  Iterator pointing to the first element to be erased.
     *  @param  last  Iterator pointing to one past the last element to be
     *                erased.
     *  @return  An iterator pointing to the element pointed to by @a last
     *           prior to erasing (or end()).
     *
     *  This function will erase the elements in the range [first,last) and
     *  shorten the %vector accordingly.
     *
     *  Note This operation could be expensive and if it is
     *  frequently used the user should consider using std::list.
     *  The user is also cautioned that this function only erases
     *  the elements, and that if the elements themselves are
     *  pointers, the pointed-to memory is not touched in any way.
     *  Managing the pointer is the user's responsibility.
     */
    iterator
    erase (iterator first, iterator last)
    {
      erase__ (first, last);
    }

    /**
     *  @brief  Resizes the %vector to the specified number of elements.
     *  @param  new_size  Number of elements the %vector should contain.
     *  @param  x  Data with which new elements should be populated.
     *
     *  This function will %resize the %vector to the specified
     *  number of elements.  If the number is smaller than the
     *  %vector's current size the %vector is truncated, otherwise
     *  the %vector is extended and new elements are populated with
     *  given data.
     */
    void
    resize (size_type new_size, const value_type &value = value_type ())
    {
      resize__ (new_size, value);
    }

    /**
     *  Erases all the elements.  Note that this function only erases the
     *  elements, and that if the elements themselves are pointers, the
     *  pointed-to memory is not touched in any way.  Managing the pointer is
     *  the user's responsibility.
     */
    void
    clear ()
    { 
      erase_at_end__ (this->size ());
    }

    /**
     *  @brief  Swaps data with another %vector.
     *  @param  x  A %vector of the same element and allocator types.
     *
     *  This exchanges the elements between two vectors in constant time.
     *  (Three pointers, so it should be quite fast.)
     *  Note that the global std::swap() function is specialized such that
     *  std::swap(v1,v2) will feed to this function.
     */
    void
    swap( shared_vector & v)
    {
      std::swap (this->array_->elems, v.array_->elems);
      std::swap (this->array_->N, v.array_->N);
      std::swap (this->capacity_, v.capacity_);
    }

    /**
     *  @brief  Assigns a given value to a %vector.
     *  @param  n  Number of elements to be assigned.
     *  @param  val  Value to be assigned.
     *
     *  This function fills a %vector with @a n copies of the given
     *  value.  Note that the assignment completely changes the
     *  %vector and that the resulting %vector's size is the same as
     *  the number of elements assigned.  Old data may be lost.
     */
    void
    assign (size_type n, const value_type &value)
    { 
      assign_fill__ (n, value); 
    }

    /**
     *  @brief  Assigns a range to a %vector.
     *  @param  first  An input iterator.
     *  @param  last   An input iterator.
     *
     *  This function fills a %vector with copies of the elements in the
     *  range [first,last).
     *
     *  Note that the assignment completely changes the %vector and
     *  that the resulting %vector's size is the same as the number
     *  of elements assigned.  Old data may be lost.
     */
    template<typename input_iterator>
    void
    assign(input_iterator first, input_iterator last)
    {
      // Check whether it's an integral type.  If so, it's not an iterator.
      typedef is_integral__ <std::numeric_limits <input_iterator>::is_integer> integral_t;
      assign_dispatch__ (first, last, integral_t ());
    }

    shared_vector ()
    : capacity_ (0)
    {
    }

    shared_vector (size_type n, const value_type &value = value_type ())
    : capacity_ (0)
    {
      ctor_fill__ (n, value);
    }

    template <typename input_iterator>
    shared_vector (input_iterator first, input_iterator last)
    : capacity_ (0)
    {
      typedef is_integral__ <std::numeric_limits <input_iterator>::is_integer> integral_t;
      ctor_dispatch__ (first, last, integral_t ());
    }

    size_type 
    capacity () const
    {
      return capacity_;
    }

    using base_t::back;

    size_type capacity_;

  private:
    size_t new_capacity (size_t i) const
    {
      return this->array_->N + std::max (this->array_->N, i);
    }

  };

} // namespace blue_sky

void
test_shared_vector ();

#endif // #ifndef BS_SHARED_VECTOR_H_

