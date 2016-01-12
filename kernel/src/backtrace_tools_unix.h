/// @file
/// @author Sergey Miryanov
/// @date 16.06.2009
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_BOS_CORE_BACKTRACE_TOOLS_UNIX_H_
#define BS_BOS_CORE_BACKTRACE_TOOLS_UNIX_H_

#include <execinfo.h>

namespace blue_sky {
namespace tools {

  static int 
  get_backtrace (void **backtrace_, int size_)
  {
    return backtrace (backtrace_, size_);
  }

  static char **
  get_backtrace_names (void *const *backtrace_, int size_)
  {
    return backtrace_symbols (backtrace_, size_);
  }

} // namespace tools
} // namespace blue_sky

#endif // #ifndef BS_BOS_CORE_BACKTRACE_TOOLS_UNIX_H_

