/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief Implementation os link
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/link.h>
#include <bs/node.h>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

namespace {

// global random UUID generator for BS links
static boost::uuids::random_generator gen;

} // eof hidden namespace

link::link(std::string name, Flags f)
	: name_(std::move(name)),
	id_(gen()), flags_(f)
{}

// copy ctor does not copy uuid from lhs
// instead it creates a new one
// [NOTE] important - owner kept empty in copied link
link::link(const link& lhs)
	: std::enable_shared_from_this<link>(lhs), name_(lhs.name_), id_(gen()), flags_(lhs.flags_), owner_()
{}

link::~link() {}

// get link's object ID
std::string link::oid() const {
	auto obj = data();
	if(obj) return obj->id();
	return boost::uuids::to_string(boost::uuids::nil_uuid());
}

std::string link::obj_type_id() const {
	auto obj = data();
	if(obj) return obj->type_id();
	return type_descriptor::nil().name;
}

void link::reset_owner(const sp_node& new_owner) {
	owner_ = new_owner;
}

link::Flags link::flags() const {
	return flags_;
}

void link::set_flags(Flags new_flags) {
	flags_ = new_flags;
}

bool link::rename(std::string new_name) {
	if(auto O = owner()) {
		return O->rename(O->find(id()), std::move(new_name));
	}
	name_ = std::move(new_name);
	return true;
}

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

