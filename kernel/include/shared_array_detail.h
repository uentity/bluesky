/**
 * \file shared_array_detail.h
 * \brief
 * \author Sergey Miryanov
 * \date 06.11.2009
 * */
#ifndef BS_SHARED_ARRAY_DETAIL_H_
#define BS_SHARED_ARRAY_DETAIL_H_

#include "throw_exception.h"

namespace blue_sky {
namespace detail {

  template <typename forward_iterator, typename size_type, typename value_type, typename allocator_t>
  void
  uninitialized_fill_n_a (forward_iterator first, size_type n, const value_type& value, allocator_t &allocator)
  {
    forward_iterator cur = first;
    try
      {
        for (; n > 0; --n, ++cur)
          allocator.construct (&*cur, value);
      }
    catch(...)
      {
        //std::_Destroy(__first, __cur, __alloc);
        //__throw_exception_again;
        bs_throw_exception ("");
      }
  }

  template <typename input_iterator, typename forward_iterator, typename allocator_t>
  forward_iterator
  uninitialized_copy_a (input_iterator first, input_iterator last, forward_iterator result, allocator_t &allocator)
  {
    forward_iterator cur = result;
    try
      {
        for (; first != last; ++first, ++cur)
          allocator.construct (&*cur, *first);

        return cur;
      }
    catch(...)
      {
        //std::_Destroy(__first, __cur, __alloc);
        //__throw_exception_again;
        bs_throw_exception ("");
      }
  }

  template <typename forward_iterator, typename allocator_t>
  void
  destroy (forward_iterator first, forward_iterator last, allocator_t &allocator)
  {
    for (; first != last; ++first)
      allocator.destroy (&*first);
  }

  template <typename pointer, typename size_type, typename allocator_t>
  void
  deallocate (pointer p, size_type n, allocator_t &allocator)
  {
    if (p)
      {
        allocator.deallocate (p, n);
      }
  }

} // namespace detail
} // namespace blue_sky

#endif // #ifndef BS_SHARED_ARRAY_DETAIL_H_

