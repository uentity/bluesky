/// @file
/// @author uentity
/// @date 05.03.2007
/// @brief Just BlueSky object base implimentation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/objbase.h>
#include <bs/kernel/types_factory.h>
#include <bs/tree/errors.h>
#include <bs/tree/inode.h>

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

// -----------------------------------------------------
// Implementation of class: object_base
// -----------------------------------------------------

NAMESPACE_BEGIN(blue_sky)

namespace {

// global random UUID generator for BS objects
static boost::uuids::random_generator gen;

} // eof hidden namespace

objbase::objbase(std::string custom_oid)
	: objbase(false, custom_oid)
{
	start_engine();
}

objbase::objbase(bool is_node, std::string custom_oid)
	: id_(custom_oid.size() ? std::move(custom_oid) : boost::uuids::to_string(gen())),
	is_node_(is_node)
{
	start_engine();
}

objbase::objbase(const objbase& obj)
	: enable_shared_from_this(obj), id_(boost::uuids::to_string(gen())), is_node_(obj.is_node_)
{
	start_engine();
}

void objbase::swap(objbase& rhs) {
	std::swap(is_node_, rhs.is_node_);
	std::swap(id_, rhs.id_);
	std::swap(inode_, rhs.inode_);
	std::swap(actor_, rhs.actor_);
}

objbase& objbase::operator=(const objbase& rhs) {
	objbase(rhs).swap(*this);
	return *this;
}

const type_descriptor& objbase::bs_type() {
	static auto td = [] {
		auto td = type_descriptor(
			identity< objbase >(), identity< nil >(),
			"objbase", "Base class of all BlueSky types", std::true_type(), std::true_type()
		);
		// add constructor from custom OID
		td.add_constructor([](const std::string& custom_oid) -> sp_obj {
			return std::make_shared<objbase>(custom_oid);
		});
		return td;
	}();

	return td;
}

const type_descriptor& objbase::bs_resolve_type() const {
	return bs_type();
}

std::string objbase::type_id() const {
	return bs_resolve_type().name;
}

std::string objbase::id() const {
	return id_;
}

bool objbase::is_node() const {
	return is_node_;
}

auto objbase::raw_actor() const -> const caf::actor& {
	return actor_;
}

auto objbase::info() const -> result_or_err<tree::inode> {
	auto I = inode_.lock();
	return I ?
		result_or_err<tree::inode>(*I) :
		tl::make_unexpected(error::quiet(tree::Error::EmptyInode));
}

NAMESPACE_END(blue_sky)

