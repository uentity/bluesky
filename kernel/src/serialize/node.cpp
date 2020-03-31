/// @file
/// @author uentity
/// @date 18.11.2019
/// @brief Node serialization impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include <bs/serialize/serialize.h>
#include <bs/serialize/tree.h>
#include "../tree/node_actor.h"

#include <cereal/types/vector.hpp>

NAMESPACE_BEGIN(blue_sky)
using namespace cereal;
using namespace tree;
using blue_sky::detail::shared;

/*-----------------------------------------------------------------------------
 *  node::node_impl
 *-----------------------------------------------------------------------------*/
namespace {
// proxy leafs view to serialize 'em as separate block or 'unit'
struct leafs_view {
	node_impl& N;

	leafs_view(const node_impl& N_) : N(const_cast<node_impl&>(N_)) {}

	template<typename Archive>
	auto save(Archive& ar) const -> void {
		ar(make_size_tag(N.links_.size()));
		// save links in custom index order
		const auto& any_order = N.links_.get<Key_tag<Key::AnyOrder>>();
		for(const auto& leaf : any_order)
			ar(leaf);
	}

	template<typename Archive>
	auto load(Archive& ar) -> void {
		std::size_t sz;
		ar(make_size_tag(sz));
		// load links in custom index order
		auto& any_order = N.links_.get<Key_tag<Key::AnyOrder>>();
		for(std::size_t i = 0; i < sz; ++i) {
			link leaf;
			ar(leaf);
			//N.insert(std::move(leaf));
			any_order.insert(any_order.end(), std::move(leaf));
		}
	}
};

} // eof hidden namespace

BSS_FCN_INL_BEGIN(serialize, node_impl)
	ar(make_nvp("leafs", leafs_view(t)));
BSS_FCN_INL_END(save, node_impl)

/*-----------------------------------------------------------------------------
 *  node
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, node)
	// properly save & restore node's group BEFORE any other actions
	if constexpr(Archive::is_saving::value) {
		// save actor group's ID
		ar(make_nvp("gid", t.gid()));
	}
	else {
		// load gid
		std::string ngid;
		ar(make_nvp("gid", ngid));
		// and start engine
		t.start_engine(ngid);
	}

	ar(
		make_nvp("objbase", base_class<objbase>(&t)),
		make_nvp("node_impl", *t.pimpl_)
	);

	// correct owner of all loaded links
	if constexpr(Archive::is_loading::value)
		t.propagate_owner();
BSS_FCN_END

BSS_FCN_EXPORT(serialize, node)

NAMESPACE_END(blue_sky)

BSS_REGISTER_TYPE(blue_sky::tree::node)

BSS_REGISTER_DYNAMIC_INIT(node)
