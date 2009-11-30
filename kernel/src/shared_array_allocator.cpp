/**
 * \file shared_array_allocator.cpp
 * \brief
 * \author Sergey Miryanov
 * \date 02.11.2009
 * */
#include "shared_vector.h"
#include "shared_array_allocator.h"

#include <vector>

namespace blue_sky {

  shared_array_manager::~shared_array_manager ()
  {
#ifdef _DEBUG
    impl_->print ();
#endif
    delete impl_;
  }
  shared_array_manager::shared_array_manager ()
  : impl_ (new impl ())
  {
  }

  shared_array_manager *
  shared_array_manager::instance ()
  {
    static shared_array_manager m_;

    return &m_;
  }

  void
  shared_array_manager::add_array (void *array, size_t size, const owner_t &owner)
  {
    if (array)
      impl_->add_array (array, size, owner);
  }

  bool
  shared_array_manager::rem_array (void *array, void *owner)
  {
    return impl_->rem_array (array, owner);
  }

  void
  shared_array_manager::change_array (void *array, void *new_memory, void *new_finish, const size_t &new_capacity)
  {
    impl_->change_array (array, new_memory, new_finish, new_capacity);
  }

  void
  shared_array_manager::change_array_end (void *array, void *new_finish)
  {
    impl_->change_array_end (array, new_finish);
  }

  template struct shared_array <uint8_t>;
  template struct shared_array <float16_t>;


} // namespace blue_sky

