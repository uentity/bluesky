/// @file
/// @author uentity
/// @date 05.03.2007
/// @brief Just BlueSky object base implimentation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/objbase.h>
#include <bs/kernel.h>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

// -----------------------------------------------------
// Implementation of class: object_base
// -----------------------------------------------------

NAMESPACE_BEGIN(blue_sky)

namespace {

// global random UUID generator for BS objects
static auto gen = boost::uuids::random_generator();

} // eof hidden namespace

objbase::objbase()
	: id_(boost::uuids::to_string(gen()))
{}

objbase::objbase(const objbase& obj)
	: enable_shared_from_this(obj), id_(boost::uuids::to_string(gen()))
{}

void objbase::swap(objbase& rhs) {
	std::swap(id_, rhs.id_);
}

objbase::~objbase() {}


const type_descriptor& objbase::bs_type() {
	static type_descriptor td(
		identity< objbase >(), identity< nil >(),
		"objbase", "Base class of all BlueSky types", std::true_type(), std::true_type()
	);
	return td;
}

int objbase::bs_register_this() const {
	return BS_KERNEL.register_instance(shared_from_this());
}

int objbase::bs_free_this() const {
	return BS_KERNEL.free_instance(shared_from_this());
}

const char* objbase::type_id() const {
	return bs_type().name.c_str();
}

const char* objbase::id() const {
	return id_.c_str();
}

NAMESPACE_END(blue_sky)

