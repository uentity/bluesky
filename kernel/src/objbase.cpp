/// @file
/// @author uentity
/// @date 05.03.2007
/// @brief Just BlueSky object base implimentation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/objbase.h>
#include <bs/uuid.h>
#include <bs/kernel/types_factory.h>
#include <bs/tree/errors.h>
#include <bs/tree/inode.h>

NAMESPACE_BEGIN(blue_sky)
/*-----------------------------------------------------------------------------
 *  objbase
 *-----------------------------------------------------------------------------*/
objbase::objbase(std::string custom_oid) {
	if(custom_oid.empty()) {
		id_ = to_string(gen_uuid());
		reset_home(id_, true);
	}
	else
		id_ = std::move(custom_oid);
}

objbase::objbase(const objbase& obj) :
	enable_shared_from_this(obj), id_(obj.id_), inode_(obj.inode_)
{}

objbase::objbase(objbase&& rhs) :
	id_(std::move(rhs.id_)), inode_(std::move(rhs.inode_))
{}

auto objbase::operator=(objbase&& rhs) -> objbase& {
	swap(rhs);
	return *this;
}

auto objbase::swap(objbase& rhs) -> void {
	using std::swap;

	swap(id_, rhs.id_);
	swap(inode_, rhs.inode_);
}

auto objbase::operator=(const objbase& rhs) -> objbase& {
	objbase(rhs).swap(*this);
	return *this;
}

auto objbase::bs_type() -> const type_descriptor& {
	static auto td = [] {
		auto td = type_descriptor(
			"objbase", &type_descriptor::nil, detail::make_assigner<objbase>(), nullptr,
			"Base class of all BS types"
		);
		// add constructor from custom OID
		td.add_constructor([](const std::string& custom_oid) -> sp_obj {
			return std::make_shared<objbase>(custom_oid);
		});
		// add default copy ctor
		td.add_copy_constructor<objbase>();
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

auto objbase::info() const -> result_or_err<tree::inode> {
	auto I = inode_.lock();
	return I ?
		result_or_err<tree::inode>(*I) :
		tl::make_unexpected(error::quiet(tree::Error::EmptyInode));
}

auto objbase::data_node() const -> tree::node {
	return tree::node::nil();
}

/*-----------------------------------------------------------------------------
 *  objnode
 *-----------------------------------------------------------------------------*/
objnode::objnode(std::string custom_oid) :
	objbase(std::move(custom_oid))
{}

objnode::objnode(tree::node N, std::string custom_oid) :
	objbase(std::move(custom_oid)), node_(std::move(N))
{}

objnode::objnode(const objnode& rhs) :
	objbase(rhs), node_(rhs.node_.clone(true))
{}

auto objnode::operator=(const objnode& rhs) -> objnode& {
	objnode(rhs).swap(*this);
	return *this;
}

auto objnode::swap(objnode& rhs) -> void {
	using std::swap;

	objbase::swap(rhs);
	swap(node_, rhs.node_);
}

auto objnode::data_node() const -> tree::node {
	return node_;
}

BS_TYPE_IMPL(objnode, objbase, "objnode", "Object with internal node")

NAMESPACE_END(blue_sky)

