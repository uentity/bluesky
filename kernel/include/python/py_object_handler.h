/// @file
/// @author Sergey Miryanov
/// @date 26.05.2009
/// @brief handles reference to PyObject. should be used (for example) as a subsriber for kernel events.
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_PY_OBJECT_HANDLER_9811dc51_fdc5_4a8b_b5b7_a05f45a2f51c_H_
#define BS_PY_OBJECT_HANDLER_9811dc51_fdc5_4a8b_b5b7_a05f45a2f51c_H_

#include <boost/python.hpp>
#include "throw_exception.h"

namespace blue_sky {
namespace tools {

  struct py_object_handler : public bs_slot
  {
    py_object_handler (PyObject *py_object_)
    : py_object_ (py_object_)
    {
      if (!py_object_)
        bs_throw_exception ("py_object is null");

      boost::python::incref (py_object_);
    }

    void
    execute (const sp_mobj &, int, const sp_obj &) const
    {
      boost::python::decref (py_object_);
    }

  private:
    PyObject *py_object_;
  };

} // namespace tools
} // namespace blue_sky


#endif // #ifndef BS_PY_OBJECT_HANDLER_9811dc51_fdc5_4a8b_b5b7_a05f45a2f51c_H_

