// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

#ifndef BS_SERIALIZE_FIXDATA_BOPO3A7S
#define BS_SERIALIZE_FIXDATA_BOPO3A7S

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/integral_constant.hpp>

#include "bs_serialize_decl.h"

namespace blue_sky {

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
	struct on_save {
		typedef typename serialize_fix_applicable< T, fixer >::on_save type;
	};
	template< class T, class fixer >
	struct on_load {
		typedef typename serialize_fix_applicable< T, fixer >::on_load type;
	};

	// op == on_save/on_load
	// result is applicable fixer type
	// or int if no fixer is found
	template< class T, class fixer_chain, template< class, class > class op >
	struct extract_fixer {

		// generic specification that goes to next recursion level
		template< class fixer_, class is_applicable >
		struct peek_fixer {
			typedef typename fixer_::next next;
			typedef typename op< T, next >::type next_applicable;
			typedef typename peek_fixer< next, next_applicable >::type type;
		};
		// boundary condition when applicable fix is found
		template< class fixer_ >
		struct peek_fixer< fixer_, boost::true_type > {
			typedef fixer_ type;
		};
		// boundary condition when full chain is processed
		template< class is_applicable >
		struct peek_fixer< int, is_applicable > {
			typedef int type;
		};

		// process fixers chain in compile-time
		typedef typename op< T, fixer_chain >::type is_applicable;
		typedef typename peek_fixer< fixer_chain, is_applicable >::type type;
	};

public:
	typedef typename Archive::is_saving is_saving;
	typedef typename Archive::is_loading is_loading;
	typedef typename serialize_first_fixer< serialize_fix_data >::type first_fixer;

	serialize_fix_data(Archive& ar) : ar_(ar) {}
	serialize_fix_data(const serialize_fix_data& rhs) : ar_(const_cast< Archive& >(rhs.ar_)) {}

	// SAVE PATH
	// go_save() actually parse chain of fixes
	// int is fixers chain terminator
	template< class Archive_, class T >
	static void go_save(Archive_& ar, const T& v, const int) {
		// if we are here then none of fixes were applied -- just plain write value
		ar << v;
	}

	template< class Archive_, class T, class fixer >
	static void go_save(Archive_& ar, const T& v, const fixer& f) {
		// if some fix sucessfully applied -- we're done
		if(do_fix_save(ar, v, f, typename serialize_fix_applicable< T, fixer >::on_save()))
			return;
		// goto next node in chain
		go_save(ar, v, typename fixer::next());
	}

	// do_fix_save() actually selects whether to apply fix and apply it
	// returns true if fix is actually applied
	// if current fix isn't applicable -- do nothing
	template< class Archive_, class T, class fixer >
	static bool do_fix_save(Archive_& ar, const T& v, const fixer&, const boost::false_type) {
		return false;
	}

	// specialization for applicable fix -- invoke fixer processing
	template< class Archive_, class T, class fixer >
	static bool do_fix_save(Archive_& ar, const T& v, const fixer&, const boost::true_type) {
		fixer::do_fix_save(ar, v);
		return true;
	}

	// overload saving operator
	template< class T >
	serialize_fix_data& operator <<(const T& v) {
		//typedef typename resolve_save_ret_t< T, first_fixer >::type R;
		//const R& fv = go_save(v, first_fixer());
		//ar_ << fv;
		go_save(ar_, v, first_fixer());
		return *this;
	}

	// LOAD PATH
	// go_load() actually parse chain of fixes
	// chain boundary operation
	template< class Archive_, class T >
	static void go_load(Archive_& ar, T& v, const int) {
		// if we are here then none of fixes were applied -- just plain read value
		ar >> v;
	}

	template< class Archive_, class T, class fixer >
	static void go_load(Archive_& ar, T& v, const fixer& f) {
		// if some fix sucessfully applied -- we're done
		if(do_fix_load(ar, v, f, typename serialize_fix_applicable< T, fixer >::on_load()))
			return;
		// goto next node in chain
		go_load(ar, v, typename fixer::next());
	}

	// do_fix_load() actually selects whether to apply fix and apply it
	template< class Archive_, class T, class fixer >
	static bool do_fix_load(Archive_& ar, T& v, const fixer&, const boost::false_type) {
		return false;
	}

	template< class Archive_, class T, class fixer >
	static bool do_fix_load(Archive_& ar, T& v, const fixer&, const boost::true_type) {
		// call fixer to load value
		fixer::do_fix_load(ar, v);
		return true;
	}

	// overload loading operator
	template< class T >
	serialize_fix_data& operator >>(T& v) {
		go_load(ar_, v, first_fixer());
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

