/**
 * \file shared_vector.h
 * \brief vector that can be shared between different objects with like std::vector facilities
 * \author Sergey Miryanov
 * \date 03.11.2009
 * */
#ifndef BS_SHARED_VECTOR_H_
#define BS_SHARED_VECTOR_H_

#include "shared_array.h"

#include <boost/type_traits.hpp>

namespace blue_sky {

  namespace detail {

    template <typename T, bool is_arithmetic>
    struct shared_vector_opt
    {
    };

    template <typename T>
    struct shared_vector_opt <T, false>
    {
      template <typename forward_iterator, typename size_type, typename value_type, typename allocator_t>
      static void
      uninitialized_fill_n_a_unsafe (forward_iterator first, size_type n, const value_type& value, allocator_t &allocator)
      {
        forward_iterator cur = first;
        for (; n > 0; --n, ++cur)
          allocator.construct (&*cur, value);
      }

      template <typename forward_iterator, typename size_type, typename value_type, typename allocator_t>
      static void
      uninitialized_fill_n_a (forward_iterator first, size_type n, const value_type& value, allocator_t &allocator)
      {
        try
          {
            uninitialized_fill_n_a_unsafe (first, n, value, allocator);  
          }
        catch(...)
          {
            //std::_Destroy(__first, __cur, __alloc);
            //__throw_exception_again;
            bs_throw_exception ("");
          }
      }

      template <typename input_iterator, typename forward_iterator, typename allocator_t>
      static forward_iterator
      uninitialized_copy_a_unsafe (input_iterator first, input_iterator last, forward_iterator result, allocator_t &allocator)
      {
        forward_iterator cur = result;

        for (; first != last; ++first, ++cur)
          allocator.construct (&*cur, *first);

        return cur;
      }

      template <typename input_iterator, typename forward_iterator, typename allocator_t>
      static forward_iterator
      uninitialized_copy_a (input_iterator first, input_iterator last, forward_iterator result, allocator_t &allocator)
      {
        try
          {
            return uninitialized_copy_a_unsafe (first, last, result, allocator);
          }
        catch(...)
          {
            //std::_Destroy(__first, __cur, __alloc);
            //__throw_exception_again;
            bs_throw_exception ("");
          }
      }

      static T *
      allocate_for_push_back_copy_a (shared_array <T> *a, 
        T *new_memory, T *new_finish, const size_t &new_capacity)
      {
        try
          {
            new_finish = uninitialized_copy_a_unsafe (a->begin (), a->end (), new_memory, a->allocator_);
          }
        catch (...)
          {
            detail_t <false>::destroy (new_memory, new_finish, a->allocator_);
            detail::deallocate (new_memory, new_capacity, a->allocator_);
            throw;
          }

        return new_finish;
      }

      static T *
      insert_fill_copy_a (shared_array <T> *a,
        T *pos, size_t n, const T &value, 
        T *new_memory, T *new_finish, const size_t &new_capacity)
      {
        try 
          {
            new_finish = uninitialized_copy_a_unsafe (a->begin (), pos, new_memory, a->allocator_);
            uninitialized_fill_n_a_unsafe (new_finish, n, value, a->allocator_);
            new_finish += n;
            new_finish = uninitialized_copy_a_unsafe (pos, a->end (), new_finish, a->allocator_);
          }
        catch (...)
          {
            detail_t <false>::destroy (new_memory, new_finish, a->allocator_);
            detail::deallocate (new_memory, new_capacity, a->allocator_);
            throw;
          }

        return new_finish;
      }

      template <typename forward_iterator>
      static T *
      insert_range_copy_a (shared_array <T> *a,
        T *pos, forward_iterator first, forward_iterator last,
        T *new_memory, T *new_finish, const size_t &new_capacity)
      {
        try 
          {
            new_finish = uninitialized_copy_a_unsafe (a->begin (), pos, new_memory, a->allocator_);
            new_finish = uninitialized_copy_a_unsafe (first, last, new_finish, a->allocator_);
            new_finish = uninitialized_copy_a_unsafe (pos, a->end (), new_finish, a->allocator_);
          }
        catch (...)
          {
            detail_t <false>::destroy (new_memory, new_finish, a->allocator_);
            detail::deallocate (new_memory, new_capacity, a->allocator_);
            throw;
          }

        return new_finish;
      }
    };

    template <typename T>
    struct shared_vector_opt <T, true>
    {
      template <typename forward_iterator, typename size_type, typename value_type, typename allocator_t>
      static void
      uninitialized_fill_n_a (forward_iterator first, size_type n, const value_type& value, allocator_t &)
      {
        std::fill_n (first, n, value);
        //for (size_t idx = 0; idx < n; ++idx)
        //  first[idx] = value;
      }

      template <typename input_iterator, typename forward_iterator, typename allocator_t>
      static forward_iterator
      uninitialized_copy_a (input_iterator first, input_iterator last, forward_iterator result, allocator_t &)
      {
        return std::copy (first, last, result);
        //forward_iterator cur = result;

        //size_t idx = 0;
        //for (; first != last; ++first, ++idx)
        //  cur[idx] = *first;

        //return cur + idx;
      }

      static T *
      allocate_for_push_back_copy_a (shared_array <T> *a,
        T *new_memory, T *, const size_t &)
      {
        return uninitialized_copy_a (a->begin (), a->end (), new_memory, a->allocator_);
      }

      static T *
      insert_fill_copy_a (shared_array <T> *a, 
        T *pos, size_t n, const T &value, 
        T *new_memory, T *, const size_t &)
      {
        T *new_finish = uninitialized_copy_a (a->begin (), pos, new_memory, a->allocator_);
        uninitialized_fill_n_a (new_finish, n, value, a->allocator_);
        new_finish += n;
        new_finish = uninitialized_copy_a (pos, a->end (), new_finish, a->allocator_);

        return new_finish;
      }

      template <typename forward_iterator>
      static T *
      insert_range_copy_a (shared_array <T> *a,
        T *pos, forward_iterator first, forward_iterator last,
        T *new_memory, T *, const size_t &)
      {
        T *new_finish = uninitialized_copy_a (a->begin (), pos, new_memory, a->allocator_);
        new_finish = uninitialized_copy_a (first, last, new_finish, a->allocator_);
        new_finish = uninitialized_copy_a (pos, a->end (), new_finish, a->allocator_);

        return new_finish;
      }
    };

    template <typename T, typename allocator_type = aligned_allocator <T, 16> >
    struct shared_vector_impl : shared_array <T, allocator_type>
    {
      typedef shared_array <T, allocator_type>        base_t;
      typedef typename base_t::value_type             value_type;
      typedef typename base_t::iterator               iterator;
      typedef typename std::allocator <T>::pointer    pointer;
      typedef allocator_type                          allocator_t;

      typedef shared_vector_opt <T, boost::is_arithmetic <T>::value>  opt_t;
      typedef detail::detail_t <boost::is_arithmetic <T>::value>      detail_t;

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

      template <typename forward_iterator>
      pointer
      allocate_and_copy__ (size_t capacity, forward_iterator first, forward_iterator last)
      {
        pointer result = allocator_.allocate (capacity);
        opt_t::uninitialized_copy_a (first, last, result, allocator_);
        return result;
      }

      void
      ctor_fill__ (size_t n, const value_type &value)
      {
        const size_t new_capacity = detail::new_capacity__ (1, n, true);
        pointer new_memory = allocator_.allocate (new_capacity);

        opt_t::uninitialized_fill_n_a (new_memory, n, value, allocator_);

        this->array_      = new_memory;
        this->array_end_  = new_memory + n;
        this->capacity_   = new_capacity;

        BS_ASSERT (this->owner_list_->size () == 1) (this->owner_list_->size ());
        change_owner (new_memory, new_memory + n, new_capacity);
      }

      void
      ctor_init__ ()
      {
        pointer new_memory = allocator_.allocate (this->capacity ());

        this->array_      = new_memory;
        this->array_end_  = new_memory;

        BS_ASSERT (this->owner_list_->size () == 1) (this->owner_list_->size ());
        change_owner (new_memory, new_memory, this->capacity ());
      }

      template <typename input_iterator>
      void
      ctor_range__ (input_iterator first, input_iterator last, std::input_iterator_tag);

      template <typename forward_iterator>
      void
      ctor_range__ (forward_iterator first, forward_iterator last, std::forward_iterator_tag)
      {
        const size_t n = std::distance (first, last);
        size_t new_capacity = detail::new_capacity__ (1, n, true);

        pointer new_memory = allocator_.allocate (new_capacity);
        pointer new_finish = opt_t::uninitialized_copy_a (first, last, new_memory, allocator_);

        this->array_      = new_memory;
        this->array_end_  = new_finish;
        this->capacity_   = new_capacity;

        BS_ASSERT (this->owner_list_->size () == 1) (this->owner_list_->size ());
        change_owner (new_memory, new_finish, new_capacity);
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

      //template <typename shared_vector>
      //void
      //ctor_copy__ (const shared_vector &x)
      //{
      //  this->array_ = allocator_.allocate (this->capacity ());
      //  this->array_end_ = this->array_ + x.size ();

      //  opt_t::uninitialized_copy_a (x.begin (), x.end (), this->begin (), allocator_);

      //  BS_ASSERT (this->owner_list_->size () == 1) (this->owner_list_->size ());
      //  this->owner_list_->push_back (this);
      //}

      void
      assign_fill__ (size_t n, const value_type &value)
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
            opt_t::uninitialized_fill_n_a (this->end (), n - this->size (), value, allocator_);
            
            this->change_owner (this->end () + n - this->size ());
          }
        else
          {
            std::fill_n (this->begin (), n, value);
            
            this->change_owner (this->begin () + n);
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
        const size_t n = std::distance (first, last);
        pointer old_memory = this->array_;
        if (n > this->capacity ())
          {
            size_t new_capacity = detail::new_capacity__ (this->size (), n, true);
            pointer new_memory (allocate_and_copy__ (new_capacity, first, last));

            detail_t::destroy (this->begin (), this->end (), allocator_);
            detail::deallocate (this->begin (), this->capacity (), allocator_); //size ()?
            this->change_owner (new_memory, new_memory + n, new_capacity);
          }
        else if (this->size () >= n)
          {
            std::copy (first, last, old_memory);
            this->change_owner (this->begin () + n);
          }
        else
          {
            forward_iterator middle = first;
            std::advance (middle, this->size ());
            std::copy (first, last, old_memory);
            pointer new_finish = opt_t::uninitialized_copy_a (middle, last, this->end (), allocator_);
            this->change_owner (this->begin () + size_t (new_finish - this->begin ()));
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
        const size_t new_capacity = detail::new_capacity__ (this->size (), 1, true);
        pointer new_memory = allocator_.allocate (new_capacity);
        pointer new_finish = new_memory;

        new_finish = opt_t::allocate_for_push_back_copy_a (this, new_memory, new_finish, new_capacity);

        detail_t::destroy (this->begin (), this->end (), allocator_);
        detail::deallocate (this->begin (), this->capacity (), allocator_);
        this->change_owner (new_memory, new_finish, new_capacity);
      }

      void
      push_back_value__ (const value_type &value)
      {
        allocator_.construct (&this->array_[this->size ()], value);
        this->change_owner (this->end () + 1);
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
      insert_fill__ (iterator pos, size_t n, const value_type &value)
      {
        if (n != 0)
          {
            if ((this->capacity () - this->size ()) >= n)
              {
                size_t elems_after = this->end () - pos;
                pointer old_finish = this->array_end_;

                if (elems_after > n)
                  {
                    opt_t::uninitialized_copy_a (old_finish - n, old_finish, old_finish, allocator_);
                    std::copy_backward (pos, old_finish - n, old_finish);
                    std::fill (pos, pos + n, value);

                    this->change_owner (this->end () + n);
                  }
                else
                  {
                    opt_t::uninitialized_fill_n_a (old_finish, n - elems_after, value, allocator_);
                    this->array_end_ += n - elems_after;
                    opt_t::uninitialized_copy_a (pos, old_finish, this->end (), allocator_);
                    this->array_end_ += elems_after;
                    std::fill (pos, old_finish, value);

                    this->change_owner (this->end ());
                  }
              }
            else
              {
                BS_ASSERT (this->is_owner ());
                const size_t new_capacity = detail::new_capacity__ (this->size (), n, true);
                pointer new_memory = allocator_.allocate (new_capacity);
                pointer new_finish = new_memory;

                new_finish = opt_t::insert_fill_copy_a (this, pos, n, value, new_memory, new_finish, new_capacity);

                detail_t::destroy (this->begin (), this->end (), allocator_);
                detail::deallocate (this->begin (), this->capacity (), allocator_);
                this->change_owner (new_memory, new_finish, new_capacity);
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
            const size_t n = std::distance (first, last);
            if ((this->capacity () - this->size ()) >= n)
              {
                const size_t elems_after = this->end () - pos;
                pointer old_finish = this->array_end_;
                if (elems_after > n)
                  {
                    opt_t::uninitialized_copy_a (old_finish - n, old_finish, old_finish, allocator_);
                    std::copy_backward (pos, old_finish - n, old_finish);
                    std::copy (first, last, pos);

                    this->change_owner (this->end () + n);
                  }
                else
                  {
                    forward_iterator middle = first;
                    std::advance (middle, elems_after);
                    opt_t::uninitialized_copy_a (middle, last, this->end (), allocator_);
                    this->array_end_ += n - elems_after;
                    opt_t::uninitialized_copy_a (pos, old_finish, this->end (), allocator_);
                    this->array_end_ += elems_after;
                    std::copy (first, middle, pos);
                    
                    this->change_owner (this->end ());
                  }
              }
            else
              {
                BS_ASSERT (this->is_owner ());
                const size_t new_capacity = detail::new_capacity__ (this->size (), n, true);
                pointer new_memory = allocator_.allocate (new_capacity);
                pointer new_finish = new_memory;
                
                new_finish = opt_t::insert_range_copy_a (this, pos, first, last, new_memory, new_finish, new_capacity);

                detail_t::destroy (this->begin (), this->end (), allocator_);
                detail::deallocate (this->begin (), this->capacity (), allocator_);
                this->change_owner (new_memory, new_finish, new_capacity);
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
        size_t n = pos - this->begin ();
        if (pos == this->end () && valid_size__ ())
          {
            push_back_value__ (value);
          }
        else
          {
            insert_fill__ (pos, size_t (1), value);
          }

        return iterator (this->begin () + n);
      }
      void
      erase_at_end__ (size_t n)
      {
        detail_t::destroy (this->end () - n, this->end (), allocator_);
        this->change_owner (this->end () - n);
      }

      iterator
      erase__ (iterator position)
      {
        if (position + 1 != this->end ())
          {
            std::copy (position + 1, this->end (), position);
          }

        detail_t::destroy (this->end () - 1, this->end (), allocator_);
        this->change_owner (this->end () - 1);
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
      resize__ (size_t new_size, const value_type &value)
      {
        if (new_size < this->size())
          erase_at_end__ (this->size () - new_size);
        else
          insert_fill__ (this->end(), new_size - this->size(), value);
      }

      //void
      //reserve__ (size_t n)
      //{
      //  if (this->capacity () < n)
      //    {
      //      size_t new_capacity = detail::new_capacity__ (allocator_, 1, n, true);
      //      pointer new_memory = this->allocate__ (new_capacity);
      //      pointer new_finish = std::copy (this->begin (), this->end (), new_memory);

      //      //detail_t::destroy (this->begin (), this->end (), allocator_);
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
        T *array_x        = this->array_;
        T *array_end_x    = this->array_end_;
        size_t capacity_x = this->capacity_;

        BS_ASSERT (this->owner_list_ && v.owner_list_);

        this->change_owner (v.array_, v.array_end_, v.capacity_);
        v.change_owner (array_x, array_end_x, capacity_x);
      }
    };
  }

  template <typename T, typename allocator_t__ = aligned_allocator <T, 16> >
  struct shared_vector : detail::shared_vector_impl <T, allocator_t__>
  {
    typedef detail::shared_vector_impl <T, allocator_t__>   base_t;
    typedef typename base_t::value_type                     value_type;
    typedef typename base_t::iterator                       iterator;
    typedef typename std::allocator <T>::pointer            pointer;
    typedef allocator_t__                                   allocator_t;
    typedef size_t                                          size_type;

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
    insert (iterator pos, size_t n, const value_type &value)
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
    resize (size_t new_size, const value_type &value = value_type ())
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
    assign (size_t n, const value_type &value)
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
    //reserve (size_t n)
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

    shared_vector (size_t n, const value_type &value = value_type (), const allocator_t &allocator = allocator_t ())
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

} // namespace blue_sky

#include "shared_array_allocator.h"

void
test_shared_vector ();

#endif // #ifndef BS_SHARED_VECTOR_H_

