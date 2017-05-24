/// @file
/// @author uentity
/// @date 03.04.2017
/// @brief any_array is an array of values of arbitrary type, based on boost::any
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "common.h"
#include <boost/any.hpp>
#include <map>

NAMESPACE_BEGIN(blue_sky)

template< template< class > class cont_traits >
class any_array : public cont_traits< boost::any > {
public:
	using container_t = cont_traits< boost::any >;

private:
	// SFINAE check if container T has mapped_type defined
	// i.e. is a map-like
	template< typename T >
	struct void_ { using type = void; };

	template< typename T, typename = void >
	struct has_mapped_t : std::false_type {};

	template< typename T >
	struct has_mapped_t< T, typename void_< typename T::mapped_type >::type > : std::true_type {};

	// trait for vector-like containers
	template< typename Cont = container_t, typename = void >
	struct trait_impl {
		using key_type = std::size_t;

		template< typename Iterator >
		static auto& iter2val(const Iterator& i) {
			return *i;
		}

		template< class Iterator >
		static key_type iter2key(const Cont& C, const Iterator& i) {
			return std::distance(C.begin(), i);
		}

		// use bound-checking to simulate map behaviour and make comparison of returned iterator
		// with C.end() valid
		template< typename Container >
		static auto find(Container& C, const key_type& k) {
			auto res = C.begin();
			std::advance(res, std::min(k, C.size()));
			return res;
		}

		static bool has_key(const Cont& C, const key_type& k) {
			return k < C.size();
		}
	};

	// trait for map-like containers
	template< typename Cont >
	struct trait_impl< Cont, typename std::enable_if_t< has_mapped_t< Cont >::value > > {
		using key_type = typename Cont::key_type;

		template< typename Iterator >
		static auto& iter2val(const Iterator& i) {
			return i->second;
		}

		template< class Iterator >
		static const key_type& iter2key(const Cont& C, const Iterator& i) {
			return i->first;
		}

		template< typename Container >
		static auto find(Container& C, const key_type& k) {
			return C.find(k);
		}

		static bool has_key(const Cont& C, const key_type& k) {
			return C.find(k) != C.end();
		}
	};

	using trait = trait_impl< container_t >;

public:

	// helper class to auto-convert return type to destination const reference type
	struct value_cast {
		explicit value_cast(const boost::any& val) : val_(val) {}

		// only move semantics allowed
		value_cast(const value_cast&) = delete;
		value_cast& operator=(const value_cast&) = delete;
		value_cast(value_cast&&) = default;

		// allow only implicit conversions to const reference
		template< typename T >
		operator T() const && {
			return boost::any_cast< const T& >(val_);
		}

	private:
		const boost::any& val_;
	};

	using key_type = typename trait::key_type;

	// import base class constructors
	using container_t::container_t;
	// import commonly used functions
	using container_t::begin;
	using container_t::end;
	using container_t::size;

	// count number of elements of givent type
	template< typename T >
	std::size_t size() const {
		std::size_t res = 0;
		for(auto pv = begin(); pv != end(); ++pv) {
			if(trait::iter2val(pv).type() == typeid(T))
				++res;
		}
		return res;
	}

	// test if array contain specified key
	bool has_key(const key_type& k) const {
		return trait::has_key(*this, k);
	}

	// return reference to value at specified index
	// value type is specified explicitly
	template< typename value_type >
	value_type& ss(const key_type& key) {
		return boost::any_cast< value_type& >(
			trait::iter2val(trait::find(*this, key))
		);
	}
	template< typename value_type >
	const value_type& ss(const key_type& key) const {
		return boost::any_cast< const value_type& >(
			trait::iter2val(trait::find(*this, key))
		);
	}

	// value type is omitted - do lazy cast
	// can only appear on right side of expression
	value_cast ss(const key_type& key) const {
		return value_cast(
			trait::iter2val(trait::find(*this, key))
		);
	}

	// tries to find value by given key
	// if value isn't found, target value is not modified
	// does not throw
	template< typename value_type >
	bool extract(const key_type& key, value_type& target) const {
		auto pval = trait::find(*this, key);
		if(pval == end()) return false;
		try {
			// copy value from source to target
			target = boost::any_cast< const value_type& >(trait::iter2val(pval));
			return true;
		}
		catch(boost::bad_any_cast) {
			return false;
		}
	}

	// Safely extract and return COPY of some value by given key. If no such value found,
	// default value is returned.
	template< typename value_type >
	value_type at(const key_type& key, const value_type def_val = value_type()) const {
		value_type res;
		if(!extract(key, res))
			res = def_val;
		return res;
	}

	// extract all keys
	std::vector< key_type > keys() const {
		std::vector< key_type > res;
		res.reserve(size());
		for(auto pv = begin(); pv != end(); ++pv) {
			res.emplace_back(trait::iter2key(*this, pv));
		}
		return res;
	}

	// extract keys of all values of specified type
	template< typename T >
	std::vector< key_type > keys() const {
		std::vector< key_type > res;
		res.reserve(size());
		for(auto pv = begin(); pv != end(); ++pv) {
			if(trait::iter2val(pv).type() == typeid(T))
				res.emplace_back(trait::iter2key(*this, pv));
		}
		return res;
	}

	// extract all values of specified type and return them in vector
	template< typename T >
	std::vector< T > values() const {
		std::vector< T > res;
		for(auto pv = begin(); pv != end(); ++pv) {
			auto& val = trait::iter2val(pv);
			if(val.type() == typeid(T))
				res.emplace_back(boost::any_cast< T >(val));
		}
		return res;
	}

	// extract all values of specified type and return them in map with corresponding keys
	template< typename T >
	auto values_map() const {
		std::map< key_type, T > res;
		for(auto pv = begin(); pv != end(); ++pv) {
			auto& val = trait::iter2val(pv);
			if(val.type() == typeid(T))
				res[trait::iter2key(*this, pv)] = boost::any_cast< T >(val);
		}
		return res;
	}

	// assign from map of key-value pairs
	// does check if key already exists in array
	// if not -- inserts new element
	template< class Map >
	any_array& operator =(const Map& rhs) {
		for(const auto& v : rhs) {
			auto pv = trait::find(*this, v.first);
			if(pv != end()) {
				trait::iter2val(pv) = v.second;
			}
			else {
				this->operator[](v.first) = v.second;
			}
		}
		return *this;
	}

};

// map-like any_array with std::string key
template< typename T > using str_any_traits = std::map< std::string, T >;
using str_any_array = any_array< str_any_traits >;

// vector-like any_array with std::size_t keys
template< typename T > using idx_any_traits = std::vector< T >;
using idx_any_array = any_array< idx_any_traits >;

NAMESPACE_END(blue_sky)

