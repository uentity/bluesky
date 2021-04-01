/// @file
/// @author uentity
/// @date 07.03.2019
/// @brief `Property` and `dictionary of properties` definition
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "property.h"
#include "meta/is_container.h"

#include <map>
#include <iterator>
#include <algorithm>
#include <stdexcept>

NAMESPACE_BEGIN(blue_sky::prop)

class propdict;

/// traits to check if type is propdict
template<typename T>
inline constexpr auto is_propdict_v = std::is_same_v<propdict, meta::remove_cvref_t<T>>;

class propdict : public std::map< std::string, property, std::less<> > {
	// traits to detect map-like classes, but not propdict
	template<typename M>
	inline static constexpr auto is_foreign_map = meta::is_map_v<M> && !is_propdict_v<M>;

public:
	// helper class to auto-convert return type to destination const reference type
	struct value_cast {
		explicit value_cast(const property& val) : val_(val) {}

		// only move semantics allowed
		value_cast(const value_cast&) = delete;
		value_cast& operator=(const value_cast&) = delete;
		value_cast(value_cast&&) = default;

		// allow only implicit conversions to const reference
		template< typename T >
		operator T() const && { return get<T>(val_); }

	private:
		const property& val_;
	};

	// [NOTE] transparent map
	using underlying_type = std::map< std::string, property, std::less<> >;

	// import base ctors
	using underlying_type::underlying_type;
	using underlying_type::operator=;

	// init from map-like container, but not propdict (don't overlap copy & move ctors)
	template< typename Map, typename = std::enable_if_t<is_foreign_map<Map>> >
	propdict(Map&& rhs) {
		*this = std::forward<Map>(rhs);
	}

	// assign via merge from map-like container, but not from propdict
	template<typename Map>
	auto operator =(Map&& rhs) -> std::enable_if_t<is_foreign_map<Map>, propdict&> {
		return merge_props(std::forward<Map>(rhs));
	}

	auto has_key(std::string_view k) const {
		return find(k) != end();
	}

	// modifying subscripting
	template<typename Value>
	decltype(auto) ss(std::string_view key, Value&& def_val = Value()) {
		return get<Value>(try_emplace(
			key_type(key), std::forward<Value>(def_val)
		).first->second);
	}

	// non-modifying (const) subscripting behaves like `std::vector` and don't check if key exists!
	// for checked search use `extract()` or `get()` functions family
	template<typename Value>
	decltype(auto) ss(std::string_view key) const {
		return get<Value>(find(key)->second);
	}
	// non-template version can appear only on RHS
	value_cast ss(std::string_view key) const {
		return value_cast(find(key)->second);
	}

	template<typename From = void, typename To = void>
	bool extract(std::string_view key, To& target) const {
		auto pval = find(key);
		return pval != end() ? extract<From>(pval->second, target) : false;
	}

	auto keys() const -> std::vector<key_type> {
		std::vector<key_type> res;
		res.reserve(size());
		for(const auto& [key, value] : *this)
			res.push_back(key);
		return res;
	}

	// extract all values of specified type and return them in vector
	template<typename T>
	auto values() const {
		std::vector<T> res;
		for(const auto& [key, value] : *this) {
			if(std::holds_alternative<T>(value))
				res.push_back( get<T>(value) );
		}
		return res;
	}

	// extract all values of specified type and return them in map with corresponding keys
	template<typename T>
	auto values_map() const {
		std::map<key_type, T> res;
		for(const auto& [key, value] : *this) {
			if(std::holds_alternative<T>(value))
				res.emplace(key, get<T>(value));
		}
		return res;
	}

	// merge data from any map-like container
	// if key already exists in propdict - replace value, if not -- insert new element
	template<typename Map>
	auto merge_props(Map&& rhs) -> propdict& {
		assert_wrong_map<Map>();

		std::for_each(std::begin(rhs), std::end(rhs),[this, hint = std::begin(*this)](auto& src_val) mutable {
			if constexpr(std::is_lvalue_reference_v<Map>)
				hint = insert_or_assign(hint, src_val.first, src_val.second);
			else
				hint = insert_or_assign(hint, src_val.first, std::move(src_val.second));
		});
		return *this;
	}

	// merge data from any map-like container
	// keeps existing values and replace only 'none'
	template<typename Map>
	auto weak_merge_props(Map&& rhs, bool replace_none = true) -> propdict& {
		assert_wrong_map<Map>();

		const auto take_value = [](auto&& kv_pair) -> decltype(auto) {
			if constexpr(std::is_lvalue_reference_v<Map>)
				return kv_pair.second;
			else
				return std::move(kv_pair.second);
		};

		std::for_each(std::begin(rhs), std::end(rhs),[&](auto& src_val) mutable {
			auto [it, is_inserted] = try_emplace(src_val.first, take_value(src_val));
			if(!is_inserted && replace_none && it->second == none())
				it->second = take_value(src_val);
		});
		return *this;
	}

private:
	template<typename Map>
	constexpr auto assert_wrong_map() {
		using PureMap = meta::remove_cvref_t<Map>;
		static_assert(meta::is_map_v<PureMap>, "Passed container is not map-like");
		static_assert(
			std::is_convertible_v<typename PureMap::key_type, key_type>,
			"Source map should have string keys"
		);
	}
};

///  formatting support
BS_API std::string to_string(const propdict& p);
BS_API std::ostream& operator <<(std::ostream& os, const propdict& x);

template<typename T, typename U>
constexpr decltype(auto) get(U&& pdict, std::string_view key) {
	auto pval = std::forward<U>(pdict).find(key);
	if(pval != std::forward<U>(pdict).end())
		return get<T>(pval->second);
	using namespace std::literals;
	throw std::out_of_range("No property with name '"s.append(key.data(), key.size()).append("'").c_str());
}

template<typename T, typename U>
constexpr decltype(auto) get_if(U* pdict, std::string_view key) noexcept {
	return get_if<T>([&]() -> decltype(&pdict->begin()->second) {
		if(pdict) {
			if(auto pval = pdict->find(key); pval != pdict->end())
				return &pval->second;
		}
		return nullptr;
	}());
}

/// Intended to appear on right side, that's why const refs
template<typename T, typename U>
constexpr auto get_or(U* pdict, std::string_view key, const T& def_value) noexcept -> const T& {
	return get_or<T>([&]() -> decltype(&pdict->begin()->second) {
		if(pdict) {
			if(auto pval = pdict->find(key); pval != pdict->end())
				return &pval->second;
		}
		return nullptr;
	}(), def_value);
}

/// tries to find value by given key
/// if value isn't found, target value is not modified
/// return if value was found and updated
template<typename From = void, typename To = void>
auto extract(const propdict& pdict, std::string_view key, To& target) noexcept {
	auto pval = pdict.find(key);
	if(pval != pdict.end())
		return extract<From>(pval->second, target);
	return false;
}

/// traits to check if type is propbook
template<typename T>
struct is_propbook : std::false_type {};

template<typename Key>
struct is_propbook<propbook<Key>> : std::true_type {};

template<typename T>
inline constexpr auto is_propbook_v = is_propbook<meta::remove_cvref_t<T>>::value;

NAMESPACE_END(blue_sky::prop)
