/// @file
/// @author uentity
/// @date 22.06.2018
/// @brief Implementation of BS tree-related serialization
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/serialize/tree.h>
#include <bs/serialize/base_types.h>
#include <cereal/types/vector.hpp>

#include "../tree/node_impl.h"

NAMESPACE_BEGIN(blue_sky)
using namespace cereal;
using namespace tree;

/*-----------------------------------------------------------------------------
 *  node::node_impl
 *-----------------------------------------------------------------------------*/
namespace {
// proxy leafs view to serialize 'em as separate block or 'unit'
struct leafs_view {
	links_container& links_;

	leafs_view(const links_container& L) : links_(const_cast<links_container&>(L)) {}

	template<typename Archive>
	auto save(Archive& ar) const -> void {
		ar(make_size_tag(links_.size()));
		// save links in custom index order
		const auto& any_order = links_.get<Key_tag<Key::AnyOrder>>();
		for(const auto& leaf : any_order)
			ar(leaf);
	}

	template<typename Archive>
	auto load(Archive& ar) -> void {
		std::size_t sz;
		ar(make_size_tag(sz));
		// load links in custom index order
		auto& any_order = links_.get<Key_tag<Key::AnyOrder>>();
		for(std::size_t i = 0; i < sz; ++i) {
			sp_link leaf;
			ar(leaf);
			any_order.insert(any_order.end(), std::move(leaf));
		}
	}
};

} // eof hidden namespace

BSS_FCN_INL_BEGIN(serialize, node::node_impl)
	ar(
		//make_nvp("handle", t.handle_),
		make_nvp("allowed_otypes", t.allowed_otypes_),
		make_nvp("leafs", leafs_view(t.links_))
	);
BSS_FCN_INL_END(save, node::node_impl)

/*-----------------------------------------------------------------------------
 *  node
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, node)
	//t.propagate_owner();
	ar(
		make_nvp("objbase", base_class<objbase>(&t)),
		make_nvp("node_impl", t.pimpl_)
	);
	// correct owner of all loaded links
	if(Archive::is_loading::value)
		t.propagate_owner();
	//node::links_container tmp;
	//ar(tmp);
BSS_FCN_END

BSS_FCN_EXPORT(serialize, node)

NAMESPACE_END(blue_sky)

BSS_REGISTER_TYPE(blue_sky::tree::node)

BSS_REGISTER_DYNAMIC_INIT(node)
