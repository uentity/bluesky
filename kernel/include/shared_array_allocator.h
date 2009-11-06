/**
 * \file shared_array_allocator.h
 * \brief
 * \author Sergey Miryanov
 * \date 02.11.2009
 * */
#ifndef BS_TOOLS_SHARED_ARRAY_ALLOCATOR_H_
#define BS_TOOLS_SHARED_ARRAY_ALLOCATOR_H_

namespace blue_sky {

  template <typename T>
  blue_sky::shared_array <T>
  internal_array (size_t N)
  {
    typedef blue_sky::shared_array <T> shared_array_t;

    return shared_array_t (typename shared_array_t::internal_deleter (), 
      typename shared_array_t::allocator_t ().allocate (N), N);
  }

  template <typename T>
  blue_sky::shared_array <T>
  numpy_array (T *e, size_t N)
  {
    return blue_sky::shared_array <T> (typename blue_sky::shared_array <T>::numpy_deleter (), e, N);
  }

} // namespace blue_sky

#endif // #ifndef BS_TOOLS_SHARED_ARRAY_ALLOCATOR_H_

