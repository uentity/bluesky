/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_SERIALIZE_FIXDATA_BOPO3A7S
#define BS_SERIALIZE_FIXDATA_BOPO3A7S

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/integral_constant.hpp>

#include "bs_serialize_decl.h"
#include "bs_conversion.h"

namespace blue_sky {

// by default fixers chain is empty (= int)
// specialize this struct with serialize_fix_data
// to specify custom fixers chain
template< class >
struct serialize_first_fixer {
	typedef int type;
};

/*-----------------------------------------------------------------
 * special proxy object to allow preprocessing of data being read/written
 *----------------------------------------------------------------*/
template< class Archive >
class BS_API_PLUGIN serialize_fix_data {
	template< class T, class fixer >
	struct op_save {
		// type is actually inherited from boost::true_type or boost::false_type
		typedef typename serialize_fix_applicable< T, fixer >::on_save type;
		// in order to be used in templates, chack conversion
		enum { is_true = BS_CONVERSION(type, boost::true_type) };
	};
	template< class T, class fixer >
	struct op_load {
		typedef typename serialize_fix_applicable< T, fixer >::on_load type;
		enum { is_true = BS_CONVERSION(type, boost::true_type) };
	};

	// op == on_save/on_load
	// result is applicable fixer type
	// or int if no fixer is found
	template< class T, class fixer_chain, template< class, class > class op >
	struct extract_fixer {

		// generic specification that goes to next recursion level
		template< class fixer_, int can_apply >
		struct peek_fixer {
			typedef typename fixer_::next next;
			enum { next_applicable = op< T, next >::is_true };
			typedef typename peek_fixer< next, next_applicable >::type type;
		};
		// boundary condition when applicable fix is found
		template< class fixer_ >
		struct peek_fixer< fixer_, 1 > {
			typedef fixer_ type;
		};
		// boundary condition when full chain is processed
		template< int can_apply >
		struct peek_fixer< int, can_apply > {
			typedef int type;
		};

		// process fixers chain in compile-time
		enum { is_applicable = op< T, fixer_chain >::is_true };
		typedef typename peek_fixer< fixer_chain, is_applicable >::type type;
	};

public:
	typedef typename Archive::is_saving is_saving;
	typedef typename Archive::is_loading is_loading;
	typedef typename serialize_first_fixer< serialize_fix_data >::type first_fixer;

	serialize_fix_data(Archive& ar) : ar_(ar) {}
	serialize_fix_data(const serialize_fix_data& rhs) : ar_(const_cast< Archive& >(rhs.ar_)) {}

	// SAVE PATH
	// do_fix_save() actually selects whether to apply fix or simply dump value
	// depending on fixer type
	// if no fixer is applicable -- just dump value
	template< class Archive_, class T >
	static void do_fix_save(Archive_& ar, const T& v, const int) {
		ar << v;
	}

	// specialization for applicable fix -- invoke fixer processing
	template< class Archive_, class T, class fixer >
	static void do_fix_save(Archive_& ar, const T& v, const fixer&) {
		fixer::do_fix_save(ar, v);
	}

	// overload saving operator
	template< class T >
	serialize_fix_data& operator <<(const T& v) {
		do_fix_save(ar_, v, typename extract_fixer< T, first_fixer, op_save >::type());
		return *this;
	}

	// LOAD PATH
	// do_fix_load() actually selects whether to apply fix or simply read value
	// depending on fixer type
	// if no fixer is applicable -- just read value
	template< class Archive_, class T >
	static void do_fix_load(Archive_& ar, T& v, const int) {
		ar >> v;
	}

	// specialization for applicable fix -- invoke fixer processing
	template< class Archive_, class T, class fixer >
	static void do_fix_load(Archive_& ar, T& v, const fixer&) {
		// call fixer to load value
		fixer::do_fix_load(ar, v);
	}

	// overload loading operator
	template< class T >
	serialize_fix_data& operator >>(T& v) {
		do_fix_load(ar_, v, typename extract_fixer< T, first_fixer, op_load >::type());
		return *this;
	}

	// if Archive::is_saving is false
	template< class T >
	void op_saveload(T& v, const boost::mpl::bool_< false >) {
		*this >> v;
	}

	// if Archive::is_saving is true
	template< class T >
	void op_saveload(const T& v, const boost::mpl::bool_< true >) {
		*this << v;
	}

	// overload & operator
	template< class T >
	serialize_fix_data& operator &(T& v) {
		op_saveload(v, typename Archive::is_saving());
		return *this;
	}

	// conversions
	operator Archive&() {
		return ar_;
	}

	operator const Archive&() const {
		return ar_;
	}

private:
	// store reference to original archive
	Archive& ar_;
};

} /* blue_sky */


#endif /* end of include guard: BS_SERIALIZE_FIXDATA_BOPO3A7S */

