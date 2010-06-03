// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

#ifndef _BS_KERNEL_TOOLS_H
#define _BS_KERNEL_TOOLS_H

#include "bs_kernel.h"

namespace blue_sky {

class BS_API kernel_tools {
public:

	static std::string print_loaded_types();

	static std::string walk_tree(bool silent = false);

	static std::string print_registered_instances();

  static std::string get_backtrace (int backtrace_depth = 16);
};

}	// blue_sky namespace

#endif

