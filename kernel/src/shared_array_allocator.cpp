/**
 * \file shared_array_allocator.cpp
 * \brief
 * \author Sergey Miryanov
 * \date 02.11.2009
 * */
#include "shared_array.h"
#include "shared_array_allocator.h"

#include <iostream>

namespace blue_sky {

  template <typename T>
  void
  shared_array <T>::internal_deleter::operator () (array_ext <T> *e)
  {
    detail::shared_array_allocator <T>::deallocate (e->data ());
    delete e;
  }

  template <typename T>
  void
  shared_array <T>::numpy_deleter::operator () (array_ext <T> *e)
  {
    delete e;
  }

  namespace detail {

    template <typename T>
    void
    shared_array_allocator <T>::deallocate (T *e)
    {
      delete [] e;
    }

    template <typename T>
    shared_array <T>
    shared_array_allocator <T>::allocate (size_t N)
    {
      T *e = new T [N];
      return blue_sky::shared_array <T> (typename shared_array <T>::internal_deleter (), e, N);
    }

    template struct shared_array_allocator <uint8_t>;
    template struct shared_array_allocator <float16_t>;

  } // namespace detail

  template struct shared_array <uint8_t>;
  template struct shared_array <float16_t>;

} // namespace blue_sky

