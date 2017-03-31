/// @file
/// @author uentity
/// @date 26.10.2016
/// @brief BS kernel tools implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel_tools.h>
#include <bs/kernel.h>
#include <sstream>

namespace blue_sky { namespace kernel_tools {
using namespace std;

std::string print_loaded_types() {
	ostringstream outs;
	kernel& k = give_kernel::Instance();
	outs << "------------------------------------------------------------------------" << endl;
	outs << "List of loaded BlueSky types {" << endl;
	kernel::plugins_enum plugins = k.loaded_plugins();
	kernel::types_enum tp;
	for(const auto& plug : plugins) {
		outs << "Plugin: [" << plug->name << "] [" << plug->description << "] [version "
			<< plug->version << "] {" << endl;
		const auto& types = k.plugin_types(*plug);
		for(const auto& t : types) {
			outs << "	[" << t.td().name << "] -> " << t.td().description << endl;
		}
		outs << "}" << endl;
	}

	outs << "} end of BlueSky types list" << endl;
	outs << "------------------------------------------------------------------------" << endl;
	return outs.str();
}

	
}} /* namespace blue_sky::kernel_tools */

