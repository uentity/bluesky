/// @file
/// @author uentity
/// @date 29.04.2016
/// @brief Plugin descriptor implementation etc
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/common.h>
#include <bs/type_descriptor.h>
//#include <bs/kernel.h>

using namespace std;

#define BS_NIL_PLUGIN_TAG "__blue_sky_nil_plugin__"

namespace blue_sky {

//------------------------------plugin_initializer----------------------------------------------------------
//plugin_initializer::plugin_initializer()
//	:
//	k(NULL)
//	// TODO: uncoment later
//	//k_(give_kernel::Instance())
//{}

//------------------------------plugin_descriptor-----------------------------------------------------------
plugin_descriptor::plugin_descriptor()
	: name(BS_NIL_PLUGIN_TAG), tag_(nil_type_info())
{}

plugin_descriptor::plugin_descriptor(const char* name_)
	: name(name_), tag_(nil_type_info())
{}

plugin_descriptor::plugin_descriptor(
	const BS_TYPE_INFO& plugin_tag, const char* name_, const char* version_,
	const char* description_, const char* py_namespace_
) :
	name(name_), version(version_), description(description_),
	py_namespace(py_namespace_), tag_(plugin_tag)
{}

bool plugin_descriptor::is_nil() const {
	return ::blue_sky::is_nil(tag_);
}

bool plugin_descriptor::operator <(const plugin_descriptor& pd) const {
	return tag_ < pd.tag_;
}

bool plugin_descriptor::operator ==(const plugin_descriptor& pd) const {
	return tag_ == pd.tag_;
}

} // eof blue_sky namespace

