/// @file
/// @author uentity
/// @date 14.08.2018
/// @brief Link-related implementation details
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/link.h>
#include <bs/tree/node.h>
#include <bs/atoms.h>

#include <caf/all.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

namespace {

// global random UUID generator for BS links
static boost::uuids::random_generator gen;

} // eof hidden namespace

/*-----------------------------------------------------------------------------
 *  link::impl
 *-----------------------------------------------------------------------------*/
struct link::impl {
	std::string name_;
	id_type id_;
	Flags flags_;
	/// contains link's metadata
	inode inode_;
	/// owner node
	std::weak_ptr<node> owner_;

	impl(std::string&& name, Flags f)
		: name_(std::move(name)),
		id_(gen()), flags_(f)
	{}

	auto rename(std::string new_name) -> void {
		name_ = std::move(new_name);
		if(auto O = owner_.lock()) {
			O->on_rename(id_);
		}
	}

};

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

