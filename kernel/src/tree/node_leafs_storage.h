/// @file
/// @author uentity
/// @date 28.01.2020
/// @brief Traits for links storage
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/node.h>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/mem_fun.hpp>

#include <algorithm>
#include <numeric>

NAMESPACE_BEGIN(blue_sky::tree)
// global alias to shorten typing
namespace mi = boost::multi_index;

/*-----------------------------------------------------------------------------
 *  links container
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN(detail)

// links are sorted by unique ID
using id_key = mi::const_mem_fun<
	link, lid_type, &link::id
>;
// and non-unique name
struct name_key {
	using result_type = std::string;
	auto operator()(const link& L) const { return L.name(unsafe); };
};
// and non-unique object ID
using oid_key = mi::const_mem_fun<
	link, std::string, &link::oid
>;
// and non-unique object type
using type_key = mi::const_mem_fun<
	link, std::string, &link::obj_type_id
>;
// and have random-access index that preserve custom items ordering
struct any_order {};

// convert from key alias -> key type
template<Key K>
struct Key_dispatch {
	using tag = id_key;
	using type = lid_type;
};
template<>
struct Key_dispatch<Key::Name> {
	using tag = name_key;
	using type = std::string;
};
template<>
struct Key_dispatch<Key::OID> {
	using tag = oid_key;
	using type = std::string;
};
template<>
struct Key_dispatch<Key::Type> {
	using tag = type_key;
	using type = std::string;
};
template<>
struct Key_dispatch<Key::AnyOrder> {
	using tag = any_order;
	using type = std::size_t;
};

NAMESPACE_END(detail)

// container that will store all node elements (links)
using links_container = mi::multi_index_container<
	link,
	mi::indexed_by<
		mi::sequenced< mi::tag<detail::any_order> >,
		mi::hashed_unique< mi::tag<detail::id_key>, detail::id_key >,
		mi::ordered_non_unique< mi::tag<detail::name_key>, detail::name_key >
	>
>;

template<Key K> using Key_tag = typename detail::Key_dispatch<K>::tag;
template<Key K> using Key_type = typename detail::Key_dispatch<K>::type;
template<Key K = Key::AnyOrder> using Index = typename links_container::index<Key_tag<K>>::type;

template<Key K = Key::AnyOrder> using iterator = typename Index<K>::iterator;
template<Key K = Key::AnyOrder> using const_iterator = typename Index<K>::const_iterator;
template<Key K = Key::ID> using insert_status = std::pair<iterator<K>, bool>;

template<Key K>
inline constexpr auto has_builtin_index_v = K != Key::OID && K != Key::Type;

constexpr auto has_builtin_index(Key K) { return K != Key::OID && K != Key::Type; }

/*-----------------------------------------------------------------------------
 *  range
 *-----------------------------------------------------------------------------*/
/// range is a pair that supports iteration
template<typename Iterator>
struct range_t : public std::pair<Iterator, Iterator> {
	using base_t = std::pair<Iterator, Iterator>;
	using base_t::base_t;

	template<Key K, typename Container>
	range_t(Container& c)
		: base_t(
			c.template get<Key_tag<K>>().begin(),
			c.template get<Key_tag<K>>().end()
		)
	{}

	range_t(const range_t&) = default;
	range_t(range_t&&) = default;
	range_t(const base_t& rhs) : base_t(rhs) {}

	auto begin() const noexcept { return this->first; }
	auto end() const noexcept { return this->second; }
	auto size() const noexcept {
		return static_cast<std::size_t>(
			std::max<typename Iterator::difference_type>( 0, std::distance(begin(), end()) )
		);
	}

	// convert range to vector of `R` by applying `extracter` to each element
	template<typename R, typename F>
	auto extract(F&& extracter) const -> std::vector<R> {
		auto res = std::vector<R>(size());
		if(!res.empty())
			std::transform(
				begin(), end(), res.begin(), std::forward<F>(extracter)
			);
		return res;
	}

	// convert range to vector of `R` by applying `extracter` to each iterator
	template<typename R, typename F>
	auto extract_it(F&& extracter) const -> std::vector<R> {
		auto res = std::vector<R>(size());
		if(!res.empty()) {
			std::size_t i = 0;
			for(auto it = begin(); it != end(); ++it)
				res[i++] = extracter(it);
		}
		return res;
	}

	template<Key K = Key::ID>
	auto extract_keys() const {
		return extract<Key_type<K>>(
			[kex = Key_tag<K>()](const auto& x) { return kex(x); }
		);
	}

	auto extract_values() const {
		return extract<link>(
			[](const auto& x) { return x; }
		);
	}
};
// deduction guide
template<typename Iterator> range_t(Iterator, Iterator) -> range_t<Iterator>;
// range aliases
template<Key K = Key::ID> using range = range_t<iterator<K>>;
template<Key K = Key::ID> using const_range = range_t<const_iterator<K>>;

NAMESPACE_END(blue_sky::tree)
