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

using namespace blue_sky;
using namespace cereal;

NAMESPACE_BEGIN(blue_sky)
/*-----------------------------------------------------------------------------
 *  node::node_impl
 *-----------------------------------------------------------------------------*/
namespace {
using namespace tree;

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

NAMESPACE_END(blue_sky)

BSS_FCN_INL_BEGIN(serialize, blue_sky::tree::node_impl)
	ar(make_nvp("leafs", leafs_view(t)));
BSS_FCN_INL_END(save, node_impl)

CEREAL_REGISTER_POLYMORPHIC_RELATION(tree::engine::impl, tree::node_impl)
CEREAL_REGISTER_TYPE_WITH_NAME(tree::node_impl, "node")

/*-----------------------------------------------------------------------------
 *  node
 *-----------------------------------------------------------------------------*/
BSS_FCN_BEGIN(serialize, node)
	// properly save & restore node's group BEFORE any other actions
	if constexpr(Archive::is_saving::value) {
		// for nil links save empty impl pointer
		auto Limpl = t ? t.pimpl_ : tree::sp_limpl{};
		ar(make_nvp("node", Limpl));
	}
	else {
		ar(make_nvp("node", t.pimpl_));
		if(t.pimpl_) {
			// impl is ready, we can start internal actor
			t.start_engine();
			// correct owner of all loaded links
			t.pimpl()->propagate_owner(t, false);
		}
		else
			t = tree::node::nil();
	}
BSS_FCN_END

BSS_FCN_EXPORT(serialize, node)

BSS_REGISTER_DYNAMIC_INIT(node)
