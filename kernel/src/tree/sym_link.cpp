/// @file
/// @author uentity
/// @date 20.11.2017
/// @brief Symbolic link implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include <bs/tree/tree.h>
#include <bs/tree/errors.h>
#include "tree_impl.h"

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

/// ctor -- pointee is specified by string path
sym_link::sym_link(std::string name, std::string path, Flags f)
	: link(std::move(name), f), path_(std::move(path))
{}
/// ctor -- pointee is specified directly - absolute path will be stored
sym_link::sym_link(std::string name, const sp_clink& src, Flags f)
	: sym_link(std::move(name), abspath(src), f)
{}

/// implement link's API
sp_link sym_link::clone(bool deep) const {
	// no deep copy support for symbolic link
	return std::make_shared<sym_link>(name(), path_, flags());
}

std::string sym_link::type_id() const {
	return "sym_link";
}

void sym_link::reset_owner(sp_node new_owner) {
	link::reset_owner(std::move(new_owner));
	// update link's status
	data_impl().map([this](const sp_obj& obj) {
		rs_reset(Req::Data, ReqStatus::OK);
		if(obj->is_node())
			rs_reset(Req::DataNode, ReqStatus::OK);
	});
}

result_or_err<sp_obj> sym_link::data_impl() const {
	// cannot dereference dangling sym link
	if(!owner()) return tl::make_unexpected(error::quiet(Error::UnboundSymLink));
	const auto src_link = detail::deref_path(path_, *this);
	return src_link ?
		result_or_err<sp_obj>(src_link->data_ex()) :
		tl::make_unexpected(error::quiet(Error::LinkExpired));
}

bool sym_link::is_alive() const {
	return bool(detail::deref_path(path_, *this));
}

/// return stored pointee path
std::string sym_link::src_path(bool human_readable) const {
	return human_readable ? convert_path(path_, bs_shared_this<sym_link>()) : path_;
}

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

