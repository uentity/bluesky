/**
 * \file shared_vector.h
 * \brief vector that can be shared between different objects with like std::vector facilities
 * \author Sergey Miryanov
 * \date 03.11.2009
 * */
#ifndef BS_SHARED_VECTOR_H_
#define BS_SHARED_VECTOR_H_

#include "shared_array.h"

#include "bs_kernel_tools.h"

namespace blue_sky {

  namespace detail {

    template <typename T, typename allocator_type = aligned_allocator <T, 16> >
    struct shared_vector_impl : shared_array <T, allocator_type>
    {
      typedef shared_array <T, allocator_type>        base_t;
      typedef typename base_t::value_type             value_type;
      typedef typename base_t::size_type              size_type;
      typedef typename base_t::iterator               iterator;
      typedef typename std::allocator <T>::pointer    pointer;
      typedef allocator_type                          allocator_t;

      template <bool b>
      struct is_integral__ 
      {
        enum {
          value = b,
        };
      };

      shared_vector_impl ()
      {
      }

      shared_vector_impl (const shared_vector_impl &v)
      : base_t (v)
      {
      }

      explicit shared_vector_impl (const base_t &v)
      : base_t (v)
      {
      }

      using base_t::allocator_;

    protected:

      //template <typename shared_vector>
      //shared_vector &
      //operator_assignment__ (shared_vector &this_, const shared_vector &x)
      //{
      //  if (&x != this)
      //    {
      //      const size_type xlen = x.size ();
      //      if (xlen > this->capacity ())
      //        {
      //          long long new_capacity = detail::new_capacity__ (allocator_, 1, xlen, true);
      //          pointer tmp (allocate_and_copy__ (new_capacity, x.begin (), x.end ()));

      //          detail::destroy (this->begin (), this->end (), allocator_);
      //          this->deallocate__ (this->begin (), this->capacity (), allocator_);

      //          this->array_ = tmp;
      //          this->array_end_ = this->array_ + xlen;
      //          this->capacity_ = new_capacity;
      //        }
      //      else if (this->size () >= xlen)
      //        {
      //          detail::destroy (std::copy (x.begin (), x.end (), this->begin ()), this->end (), allocator_);
      //        }
      //      else
      //        {
      //          std::copy (x.begin (), x.begin () + this->size (), this->begin ());
      //          this->array_end_ = detail::uninitialized_copy_a (x.begin () + this->size (), x.end (), this->end (), allocator_);
      //        }
      //    }

      //  return this_;
      //}
      template <typename forward_iterator>
      pointer
      allocate_and_copy__ (long long capacity, forward_iterator first, forward_iterator last)
      {
        pointer result = allocator_.allocate (capacity);
        detail::uninitialized_copy_a (first, last, result, allocator_);
        return result;
      }

      void
      ctor_fill__ (size_type n, const value_type &value)
      {
        long long new_capacity = detail::new_capacity__ (allocator_, 1, n, true);
        pointer new_memory = allocator_.allocate (new_capacity);

        detail::uninitialized_fill_n_a (new_memory, n, value, allocator_);

        this->array_      = new_memory;
        this->array_end_  = this->array_ + n;
        this->capacity_   = new_capacity;

        detail::add_owner__ (allocator_, this->begin (), new_capacity, this);
      }

      void
      ctor_init__ ()
      {
        pointer new_memory = allocator_.allocate (this->capacity ());

        this->array_      = new_memory;
        this->array_end_  = new_memory;

        detail::add_owner__ (allocator_, this->begin (), this->capacity (), this);
      }

      template <typename input_iterator>
      void
      ctor_range__ (input_iterator first, input_iterator last, std::input_iterator_tag);

      template <typename forward_iterator>
      void
      ctor_range__ (forward_iterator first, forward_iterator last, std::forward_iterator_tag)
      {
        const size_type n = std::distance (first, last);
        long long new_capacity = detail::new_capacity__ (allocator_, 1, n, true);

        pointer new_memory = allocator_.allocate (new_capacity);
        pointer new_finish = detail::uninitialized_copy_a (first, last, new_memory, allocator_);

        this->array_      = new_memory;
        this->array_end_  = new_finish;
        this->capacity_   = new_capacity;

        detail::add_owner__ (allocator_, this->begin (), new_capacity, this);
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

      template <typename shared_vector>
      void
      ctor_copy__ (const shared_vector &x)
      {
        this->array_ = allocator_.allocate (this->capacity ());
        this->array_end_ = this->array_ + x.size ();
        detail::uninitialized_copy_a (x.begin (), x.end (), this->begin (), allocator_);
        detail::add_owner__ (allocator_, this->begin (), this->capacity (), this);
      }

      void
      assign_fill__ (size_type n, const value_type &value)
      {
        if (n > this->capacity ())
          {
            shared_vector_impl tmp;
            tmp.ctor_fill__ (n, value);
            tmp.swap__ (*this);
          }
        else if (n > this->size ())
          {
            std::fill (this->begin (), this->end (), value);
            detail::uninitialized_fill_n_a (this->end (), n - this->size (), value, allocator_);
            //this->array_end_ += (n - this->size ());
            detail::change_ownee__ (allocator_, this->begin (), this->end () + n - this->size ());
          }
        else
          {
            std::fill_n (this->begin (), n, value);
            //this->array_end_ = this->array_ + n;
            detail::change_ownee__ (allocator_, this->begin (), this->begin () + n);
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
        pointer old_memory = this->array_;
        pointer old_memory_end = this->array_end_;
        if (n > this->capacity ())
          {
            long long new_capacity = detail::new_capacity__ (allocator_, this->size (), n, true);
            pointer new_memory (allocate_and_copy__ (new_capacity, first, last));

            detail::destroy (this->begin (), this->end (), allocator_);
            detail::deallocate (this->begin (), this->capacity (), allocator_); //size ()?
            detail::change_ownee__ (allocator_, this->begin (), new_memory, new_memory + n, new_capacity);

            // already set by change_ownee__
            //this->array_      = new_memory;
            //this->array_end_  = this->array_ + n;
          }
        else if (this->size () >= n)
          {
            std::copy (first, last, old_memory);
            detail::change_ownee__ (allocator_, this->begin (), this->begin () + n);
            // already set by change_ownee__
            //this->array_end_ = this->array_ + n;
          }
        else
          {
            forward_iterator middle = first;
            std::advance (middle, this->size ());
            std::copy (first, last, old_memory);
            pointer new_finish = detail::uninitialized_copy_a (middle, last, this->end (), allocator_);
            detail::change_ownee__ (allocator_, this->begin (), this->begin () + size_type (new_finish - this->begin ()));
            // already set by change_ownee__
            //this->array_end_ = this->array_ + size_type (new_finish - this->begin ());
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
      allocate_for_push_back__ ()
      {
        BS_ASSERT (this->is_owner ());
        size_type new_capacity = detail::new_capacity__ (allocator_, this->size (), 1, true);
        pointer new_memory = allocator_.allocate (new_capacity);
        pointer new_finish = new_memory;

        try
          {
            new_finish = detail::uninitialized_copy_a (this->begin (), this->end (), new_memory, allocator_);
          }
        catch (...)
          {
            detail::destroy (new_memory, new_finish, allocator_);
            detail::deallocate (new_memory, new_capacity, allocator_);
            throw;
          }

        detail::destroy (this->begin (), this->end (), allocator_);
        detail::deallocate (this->begin (), this->capacity (), allocator_);
        detail::change_ownee__ (allocator_, this->begin (), new_memory, new_finish, new_capacity);

        // already set by change_ownee__
        //this->array_      = new_memory;
        //this->array_end_  = new_finish;
        //this->capacity_   = new_capacity;
      }

      void
      push_back_value__ (const value_type &value)
      {
        allocator_.construct (&this->array_[this->size ()], value);
        detail::change_ownee__ (allocator_, this->begin (), this->end () + 1);
        // already set by change_ownee__
        //++this->array_end_;
      }

      void
      push_back__ (const value_type &value)
      {
        if (this->size () == this->capacity ())
          {
            allocate_for_push_back__ ();
          }

        push_back_value__ (value);
      }

      bool
      valid_size__ ()
      {
        return this->size () != this->capacity ();
      }

      void
      insert_fill__ (iterator pos, size_type n, const value_type &value)
      {
        if (n != 0)
          {
            if ((this->capacity () - this->size ()) >= n)
              {
                size_type elems_after = this->end () - pos;
                pointer old_finish = this->array_end_;

                if (elems_after > n)
                  {
                    detail::uninitialized_copy_a (old_finish - n, old_finish, old_finish, allocator_);
                    std::copy_backward (pos, old_finish - n, old_finish);
                    std::fill (pos, pos + n, value);

                    detail::change_ownee__ (allocator_, this->begin (), this->end () + n);
                    // already set by change_ownee__
                    //this->array_end_ += n;
                  }
                else
                  {
                    detail::uninitialized_fill_n_a (old_finish, n - elems_after, value, allocator_);
                    this->array_end_ += n - elems_after;
                    detail::uninitialized_copy_a (pos, old_finish, this->end (), allocator_);
                    this->array_end_ += elems_after;
                    std::fill (pos, old_finish, value);

                    // already set by change_ownee__
                    detail::change_ownee__ (allocator_, this->begin (), this->end ());
                  }
              }
            else
              {
                BS_ASSERT (this->is_owner ());
                size_type new_capacity = detail::new_capacity__ (allocator_, this->size (), n, true);
                pointer new_memory = allocator_.allocate (new_capacity);
                pointer new_finish = new_memory;

                try 
                  {
                    new_finish = detail::uninitialized_copy_a (this->begin (), pos, new_memory, allocator_);
                    detail::uninitialized_fill_n_a (new_finish, n, value, allocator_);
                    new_finish += n;
                    new_finish = detail::uninitialized_copy_a (pos, this->end (), new_finish, allocator_);
                  }
                catch (...)
                  {
                    detail::destroy (new_memory, new_finish, allocator_);
                    detail::deallocate (new_memory, new_capacity, allocator_);
                    throw;
                  }

                detail::destroy (this->begin (), this->end (), allocator_);
                detail::deallocate (this->begin (), this->capacity (), allocator_);
                detail::change_ownee__ (allocator_, this->begin (), new_memory, new_finish, new_capacity);

                // already set by change_ownee__
                //this->array_      = new_memory;
                //this->array_end_  = new_finish;
                //this->capacity_   = new_capacity;
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
            if ((this->capacity () - this->size ()) >= n)
              {
                const size_type elems_after = this->end () - pos;
                pointer old_finish = this->array_end_;
                if (elems_after > n)
                  {
                    detail::uninitialized_copy_a (old_finish - n, old_finish, old_finish, allocator_);
                    std::copy_backward (pos, old_finish - n, old_finish);
                    std::copy (first, last, pos);

                    detail::change_ownee__ (allocator_, this->begin (), this->end () + n);
                    // already set by change_ownee__
                    //this->array_end_ += n;
                  }
                else
                  {
                    forward_iterator middle = first;
                    std::advance (middle, elems_after);
                    detail::uninitialized_copy_a (middle, last, this->end (), allocator_);
                    this->array_end_ += n - elems_after;
                    detail::uninitialized_copy_a (pos, old_finish, this->end (), allocator_);
                    this->array_end_ += elems_after;
                    std::copy (first, middle, pos);
                    
                    detail::change_ownee__ (allocator_, this->begin (), this->end ());
                  }
              }
            else
              {
                BS_ASSERT (this->is_owner ());
                size_type new_capacity = detail::new_capacity__ (allocator_, this->size (), n, true);
                pointer new_memory = allocator_.allocate (new_capacity);
                pointer new_finish = new_memory;
                
                try 
                  {
                    new_finish = detail::uninitialized_copy_a (this->begin (), pos, new_memory, allocator_);
                    new_finish = detail::uninitialized_copy_a (first, last, new_finish, allocator_);
                    new_finish = detail::uninitialized_copy_a (pos, this->end (), new_finish, allocator_);
                  }
                catch (...)
                  {
                    detail::destroy (new_memory, new_finish, allocator_);
                    detail::deallocate (new_memory, new_capacity, allocator_);
                    throw;
                  }

                detail::destroy (this->begin (), this->end (), allocator_);
                detail::deallocate (this->begin (), this->capacity (), allocator_);
                detail::change_ownee__ (allocator_, this->begin (), new_memory, new_finish, new_capacity);

                // already set by change_ownee__
                //this->array_      = new_memory;
                //this->array_end_  = new_finish;
                //this->capacity_   = new_capacity;
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

      iterator
      insert__ (iterator pos, const value_type &value)
      {
        size_type n = pos - this->begin ();
        if (pos == this->end () && valid_size__ ())
          {
            push_back_value__ (value);
          }
        else
          {
            insert_fill__ (pos, size_type (1), value);
          }

        return iterator (this->begin () + n);
      }
      void
      erase_at_end__ (size_type n)
      {
        detail::destroy (this->end () - n, this->end (), allocator_);
        detail::change_ownee__ (allocator_, this->begin (), this->end () - n);
        // already set by change_ownee__
        //this->array_end_ -= n;
      }

      iterator
      erase__ (iterator position)
      {
        if (position + 1 != this->end ())
          {
            std::copy (position + 1, this->end (), position);
          }

        // already set by change_ownee__
        //--this->array_end_;
        detail::destroy (this->end () - 1, this->end (), allocator_);
        detail::change_ownee__ (allocator_, this->begin (), this->end () - 1);
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

      //void
      //reserve__ (size_type n)
      //{
      //  if (this->capacity () < n)
      //    {
      //      long long new_capacity = detail::new_capacity__ (allocator_, 1, n, true);
      //      pointer new_memory = this->allocate__ (new_capacity);
      //      pointer new_finish = std::copy (this->begin (), this->end (), new_memory);

      //      //detail::destroy (this->begin (), this->end (), allocator_);
      //      this->deallocate__ (this->begin (), this->capacity (), allocator_);

      //      this->array_      = new_memory;
      //      this->array_end_  = new_finish;
      //      this->capacity_   = new_capacity;
      //    }
      //}

      template <typename shared_vector>
      void
      swap__ (shared_vector &v)
      {
        detail::add_owner__ (allocator_, v.array_, v.capacity_, this);
        detail::add_owner__ (allocator_, this->array_, this->capacity_, &v);

        detail::rem_owner__ (allocator_, this->array_, this);
        detail::rem_owner__ (allocator_, v.array_, &v);

        std::swap (this->array_,     v.array_);
        std::swap (this->array_end_, v.array_end_);
        std::swap (this->capacity_,  v.capacity_);

        std::cout << "swap: " << this->array_ << " with " << v.array_ << std::endl;
      }
    };
  }

  template <typename T, typename allocator_t__ = aligned_allocator <T, 16> >
  struct shared_vector : detail::shared_vector_impl <T, allocator_t__>
  {
    typedef detail::shared_vector_impl <T, allocator_t__>   base_t;
    typedef typename base_t::value_type                     value_type;
    typedef typename base_t::size_type                      size_type;
    typedef typename base_t::iterator                       iterator;
    typedef typename std::allocator <T>::pointer            pointer;
    typedef allocator_t__                                   allocator_t;

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
      if (this->is_owner ())
        {
          push_back__ (value);
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
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
      if (this->is_owner ())
        {
          return insert__ (pos, value);
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
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
      if (this->is_owner ())
        {
          insert_fill__ (pos, n, value);
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
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
      if (this->is_owner ())
        {
          typedef typename base_t::template is_integral__ <std::numeric_limits <input_iterator>::is_integer> integral_t;
          insert_dispatch__ (position, first, last, integral_t ());
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
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
      if (this->is_owner ())
        {
          return erase__ (position);
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
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
      if (this->is_owner ())
        {
          return erase__ (first, last);
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
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
      if (this->is_owner ())
        {
          resize__ (new_size, value);
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
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
      if (this->is_owner ())
        {
          erase_at_end__ (this->size ());
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
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
    swap (shared_vector &v)
    {
      if (this->is_owner ())
        {
          swap__ (v);
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
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
      if (this->is_owner ())
        {
          assign_fill__ (n, value); 
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
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
      if (this->is_owner ())
        {
          typedef typename base_t::template is_integral__ <std::numeric_limits <input_iterator>::is_integer> integral_t;
          assign_dispatch__ (first, last, integral_t ());
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
    }

    /**
     *  @brief  %Vector assignment operator.
     *  @param  x  A %vector of identical element and allocator types.
     *
     *  All the elements of @a x are copied, but any extra memory in
     *  @a x (for fast expansion) will not be copied.  Unlike the
     *  copy constructor, the allocator object is not copied.
     */
    //shared_vector &
    //operator= (const shared_vector &x)
    //{
    //  this->array_ = x.array_;
    //  this->owned_ = x.owned_;

    //  return *this;
    //  //if (this->owned_)
    //  //  {
    //  //    return operator_assignment__ (*this, x);
    //  //  }
    //  //else
    //  //  {
    //  //    bs_throw_exception ("Error: shared_vector not owns data");
    //  //  }
    //}

    /**
     *  @brief  Attempt to preallocate enough memory for specified number of
     *          elements.
     *  @param  n  Number of elements required.
     *  @throw  std::length_error  If @a n exceeds @c max_size().
     *
     *  This function attempts to reserve enough memory for the
     *  %vector to hold the specified number of elements.  If the
     *  number requested is more than max_size(), length_error is
     *  thrown.
     *
     *  The advantage of this function is that if optimal code is a
     *  necessity and the user can determine the number of elements
     *  that will be required, the user can reserve the memory in
     *  %advance, and thus prevent a possible reallocation of memory
     *  and copying of %vector data.
     */
    //void
    //reserve (size_type n)
    //{
    //  if (this->is_owner ())
    //    {
    //      reserve__ (n);
    //    }
    //  else
    //    {
    //      bs_throw_exception ("Error: shared_vector not owns data");
    //    }
    //}

    shared_vector ()
    {
      this->ctor_init__ ();
    }

    shared_vector (size_type n, const value_type &value = value_type (), const allocator_t &allocator = allocator_t ())
    {
      if (this->is_owner ())
        {
          ctor_fill__ (n, value);
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
    }

    template <typename input_iterator>
    shared_vector (input_iterator first, input_iterator last)
    {
      if (this->is_owner ())
        {
          typedef typename base_t::template is_integral__ <std::numeric_limits <input_iterator>::is_integer> integral_t;
          ctor_dispatch__ (first, last, integral_t ());
        }
      else
        {
          bs_throw_exception ("Error: shared_vector not owns data");
        }
    }

    shared_vector (const shared_vector &x)
    : base_t (x)
    {
      //if (this->owned_)
      //  {
      //    ctor_copy__ (x);
      //  }
      //else
      //  {
      //    bs_throw_exception ("Error: shared_vector not owns data");
      //  }
    }

    explicit shared_vector (const shared_array <T, allocator_t> &x)
    : base_t (x)
    {
    }

    using base_t::back;
    using base_t::assign;
    using base_t::size;
  };

  typedef unsigned char               uint8_t;
  typedef float                       float16_t;

  typedef shared_vector <uint8_t>     array_uint8_t;
  typedef shared_vector <float16_t>   array_float16_t;

  template <typename T>
  struct shared_array_manager <T>::impl
  {
    typedef shared_array <T, aligned_allocator <T, 16> > shared_array_t;

    void
    add_array (T *array, size_t size, void *owner)
    {
      shared_array_t *sa = static_cast <shared_array_t *> (owner);

      for (size_t i = 0, cnt = arrays_.size (); i < cnt; ++i)
        {
          array_info &info = arrays_[i];
          if (info.array_ == array)
            {
              info.owners_.push_back (sa);
              //std::cout << "Add new owner " << owner << " to " << array << std::endl;
              //std::cout << kernel_tools::get_backtrace (32) << std::endl;
              //if (size != info.size_)
              //  {
              //    std::cout << "Size mismatch" << std::endl;
              //  }

              return ;
            }
        }

      array_info info;
      info.array_ = array;
      info.size_ = size;
      info.owners_.push_back (sa);
      arrays_.push_back (info);

      //std::cout << "Add owner " << owner << " to array " << (void *)array << std::endl;
      //std::cout << kernel_tools::get_backtrace (32) << std::endl;
    }

    bool
    rem_array (T *array, void *owner)
    {
      shared_array_t *sa = static_cast <shared_array_t *> (owner);

      //std::cout << "Try to remove owner " << owner << " from array " << (void *)array << std::endl;
      for (size_t i = 0, cnt = arrays_.size (); i < cnt; ++i)
        {
          array_info &info = arrays_[i];
          if (info.array_ == array)
            {
              rem_owner (info.owners_, sa);
              //std::cout << "Remove owner " << owner << " from array " << (void *)array << std::endl;

              if (info.owners_.empty ())
                {
                  //std::cout << "Deallocate memory " << (void *)array << std::endl;
                  arrays_.erase (arrays_.begin () + i);
                  return true;
                }
            }
        }

      return false;
    }

    template <typename Y>
    void
    rem_owner (std::vector <Y *>  &owners, Y *owner)
    {
      for (size_t i = 0, cnt = owners.size (); i < cnt; ++i)
        {
          if (owners[i] == owner)
            {
              owners.erase (owners.begin () + i);
              break;
            }
        }
    }

    void
    change_array (T *array, T *new_memory, T *new_finish, const long long &new_capacity)
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
                  shared_array_t *sa = info.owners_[j];
                  sa->array_      = new_memory;
                  sa->array_end_  = new_finish;
                  sa->capacity_   = new_capacity;
                }

              break;
            }
        }
    }

    void
    change_array_end (T *array, T *new_finish)
    {
      for (size_t i = 0, cnt = arrays_.size (); i < cnt; ++i)
        {
          array_info &info = arrays_[i];
          if (info.array_ == array)
            {
              for (size_t j = 0, jcnt = info.owners_.size (); j < jcnt; ++j)
                {
                  shared_array_t *sa = info.owners_[j];
                  sa->array_end_  = new_finish;
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
              std::cout << "\t" << (void *)info.owners_[j] << std::endl;
            }
        }
    }

    struct array_info
    {
      T                               *array_;
      size_t                          size_;
      std::vector <shared_array_t *>  owners_;
    };

    std::vector <array_info> arrays_;
  };

  template <typename T>
  shared_array_manager <T>::shared_array_manager ()
  : impl_ (new impl ())
  {
  }
  template <typename T>
  void
  shared_array_manager <T>::add_array (T *array, size_t size, void *owner)
  {
    if (array)
      impl_->add_array (array, size, owner);
  }

  template <typename T>
  bool
  shared_array_manager <T>::rem_array (T *array, void *owner)
  {
    return impl_->rem_array (array, owner);
  }

  template <typename T>
  void
  shared_array_manager <T>::change_array (T *array, T *new_memory, T *new_finish, const long long &new_capacity)
  {
    impl_->change_array (array, new_memory, new_finish, new_capacity);
  }

  template <typename T>
  void
  shared_array_manager <T>::change_array_end (T *array, T *new_finish)
  {
    impl_->change_array_end (array, new_finish);
  }

  template <typename T>
  shared_array_manager <T>::~shared_array_manager ()
  {
    impl_->print ();
    delete impl_;
  }

  template <typename T>
  shared_array_manager <T> *
  shared_array_manager <T>::instance ()
  {
    static shared_array_manager <T> m_;

    return &m_;
  }
} // namespace blue_sky

#include "shared_array_allocator.h"

void
test_shared_vector ();

#endif // #ifndef BS_SHARED_VECTOR_H_

