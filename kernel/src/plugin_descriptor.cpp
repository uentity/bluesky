/// @file
/// @author uentity
/// @date 29.04.2016
/// @brief Plugin descriptor implementation etc
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/common.h>
#include <bs/defaults.h>
#include <bs/type_descriptor.h>
#include <bs/plugin_descriptor.h>

namespace blue_sky {
using defaults::kernel::nil_plugin_tag;

//------------------------------plugin_descriptor-----------------------------------------------------------
plugin_descriptor::plugin_descriptor()
	: name(nil_plugin_tag),
	serial_input_bindings(nullptr), serial_output_bindings(nullptr), serial_polycasters(nullptr),
	tag_(nil_type_info())
{}

plugin_descriptor::plugin_descriptor(
	const BS_TYPE_INFO& plugin_tag, const char* name_, const char* version_,
	const char* description_, const char* py_namespace_,
	void* serial_input_bindings_, void* serial_output_bindings_,
	void* serial_polycasters_
) :
	name(name_), version(version_), description(description_), py_namespace(py_namespace_),
	serial_input_bindings(serial_input_bindings_), serial_output_bindings(serial_output_bindings_),
	serial_polycasters(serial_polycasters_),
	tag_(plugin_tag)
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

const plugin_descriptor& plugin_descriptor::nil() {
	static plugin_descriptor nil_pd(nil_type_info(), nil_plugin_tag, "");
	return nil_pd;
}

} // eof blue_sky namespace

