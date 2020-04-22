/// @file
/// @author uentity
/// @date 29.06.2018
/// @brief Sample objects for testing
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <bs/objbase.h>
#include <bs/tree/link.h>
#include <bs/serialize/serialize.h>

NAMESPACE_BEGIN(blue_sky)

///////////////////////////////////////////////////////////////////////////////
// [NOTE] `type_descriptor` implementation for all these classes can be found in
//  test_type_descriptor.cpp 
//
/*-----------------------------------------------------------------------------
 * non-templated class
 *-----------------------------------------------------------------------------*/
class bs_person : public objbase {
public:
	bs_person() : name_("[NONAME]"), age_(0) {}

	bs_person(const char* name) : name_(name), age_(0) {}

	bs_person(double age) : name_("[NONAME]"), age_(age) {}

	bs_person(const std::string& name, double age = 0) : name_(name), age_(age) {}

	template< class Ostream >
	friend Ostream& operator<<(Ostream& os, const bs_person& p) {
		return os << "Person name = " << p.name_ << ", age = " << p.age_;
	}

	std::string name_;
	double age_;

	BS_TYPE_DECL
};
using sp_person = std::shared_ptr< bs_person >;

BSS_FCN_DECL(serialize, bs_person)

auto make_persons_tree() -> tree::link;

/*-----------------------------------------------------------------------------
 * templated class
 *-----------------------------------------------------------------------------*/
template< class T >
struct my_strategy : public objbase {
	using cont_type = std::vector< T >;

	my_strategy() : name_("[NONAME]") {}
	my_strategy(const my_strategy&) = default;
	my_strategy(my_strategy&&) = default;
	my_strategy(const std::string& name) : name_(name) {}

	std::string name_;

	//template< class Ostream >
	friend std::ostream& operator<<(std::ostream& os, const my_strategy& s) {
		return os << "Strategy type : " << bs_type().name << "; strategy name = " << s.name_;
	}

	// expect that only one strategy can exist
	BS_TYPE_DECL_INL(my_strategy, objbase, "", "Strategy")
};
template< class T >
using sp_strat = std::shared_ptr< my_strategy< T > >;

BSS_FCN_DECL_T(serialize, my_strategy, 1)

/*-----------------------------------------------------------------------------
 * complex templated class
 *-----------------------------------------------------------------------------*/
// Strategy is passed with tuple argument just for testing
template< class T, class Strategy >
class uber_type : public objbase {
public:
	using uber_T = T;
	using cont_T = typename Strategy::cont_type;

	uber_type() : name_("[NONAME]") {};
	uber_type(const uber_type&) = default;
	uber_type(const char* name) : name_(name) {}
	uber_type(const std::string& name) : name_(name) {}

	void add_value(uber_T val) { storage_.emplace_back(std::move(val)); }

	//template< class Ostream >
	friend std::ostream& operator<<(std::ostream& os, const uber_type& s) {
		os << "Uber type: " << bs_type().name << "; Uber name = " << s.name_ <<
			" has elements [" << s.storage_.size() << "]: ";
		for(auto& v : s.storage_) {
			os << v << ' ';
		}
		return os;
	}

	T value_;
	cont_T storage_;
	std::string name_;

	BS_TYPE_DECL_INL_BEGIN(uber_type, objbase, "", "Uber complex type")
		td.add_constructor< uber_type, const char* >();
		td.add_copy_constructor< uber_type >();
	BS_TYPE_DECL_INL_END
};
template< class T >
using uber_t = uber_type< T, my_strategy< T > >;
template< class T >
using sp_uber = std::shared_ptr< uber_t< T > >;

BSS_FCN_DECL_T(serialize, uber_type, 2)

NAMESPACE_END(blue_sky)

BSS_REGISTER_TYPE(blue_sky::bs_person)

BSS_FORCE_DYNAMIC_INIT(test_objects)

