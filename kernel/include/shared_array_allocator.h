/**
 * \file shared_array_allocator.h
 * \brief
 * \author Sergey Miryanov
 * \date 02.11.2009
 * */
#ifndef BS_TOOLS_SHARED_ARRAY_ALLOCATOR_H_
#define BS_TOOLS_SHARED_ARRAY_ALLOCATOR_H_

namespace blue_sky {

  namespace detail {

    template <typename T>
    struct BS_API_PLUGIN shared_array_allocator
    {
      static blue_sky::shared_array <T>
      allocate (size_t N);

      static void
      deallocate (T *e);
    };
  }


  template <typename T>
  blue_sky::shared_array <T>
  internal_array (size_t N)
  {
    return detail::shared_array_allocator <T>::allocate (N);
  }

  template <typename T>
  blue_sky::shared_array <T>
  numpy_array (T *e, size_t N)
  {
    return blue_sky::shared_array <T> (typename blue_sky::shared_array <T>::numpy_deleter (), e, N);
  }

} // namespace blue_sky

#endif // #ifndef BS_TOOLS_SHARED_ARRAY_ALLOCATOR_H_

