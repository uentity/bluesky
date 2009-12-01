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
  blue_sky::shared_vector <T, aligned_allocator <T, 16> >
  numpy_array (T *e, size_t N)
  {
    typedef blue_sky::shared_array <T, aligned_allocator <T, 16> > shared_array_t;
    typedef blue_sky::shared_vector <T, aligned_allocator <T, 16> > shared_vector_t;

    return shared_vector_t (shared_array_t (typename shared_array_t::numpy_deleter (), e, N));
  }

} // namespace blue_sky

#endif // #ifndef BS_TOOLS_SHARED_ARRAY_ALLOCATOR_H_

