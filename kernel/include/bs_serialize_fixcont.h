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

#ifndef BS_SERIALIZE_FIXCONT_W1TRZ4NG
#define BS_SERIALIZE_FIXCONT_W1TRZ4NG

#include "bs_serialize_fixdata.h"
#include <iterator>
#include <vector>
#include <deque>
#include <list>
#include <set>
#include <map>

namespace blue_sky {
/*-----------------------------------------------------------------
 * apply nested converter to each container element on serialize
 *----------------------------------------------------------------*/
// MUST be used with non-empty nested fix (next_fix)
// idea is that we invoke processing filters for each container element
// my manually serializing them using functions from serialize_fixdata< Archive >
template< class next_fix >
struct serialize_fix_cont {
	typedef next_fix next;

	// SAVE PATH
	// this version should be appropriate for all non-map containers
	template< class Archive, class cont_t >
	static void do_fix_save(Archive& ar, const cont_t& v) {
		serialize_fix_data< Archive > fixar(ar);

		// first of all, save container size
		fixar << (const std::size_t&)v.size();
		// for each element invoke next_fix processor
		for(typename cont_t::const_iterator i = v.begin(), end = v.end(); i != end; ++i)
			fixar << *i;
	}

	// save std::map
	template< class Archive, class map_t >
	static void save_map(Archive& ar, const map_t& v) {
		serialize_fix_data< Archive > fixar(ar);

		// first of all, save map size
		fixar << (const std::size_t&)v.size();
		// for each element
		for(typename map_t::const_iterator i = v.begin(), end = v.end(); i != end; ++i) {
			// we need to process both keys and values
			fixar << i->first << i->second;
		}
	}

	// specialization for std::map
	template< class Archive, class Key, class Data, class Compare, class Alloc >
	static void do_fix_save(Archive& ar, const std::map< Key, Data, Compare, Alloc >& v) {
		save_map(ar, v);
	}
	// and std::multimap
	template< class Archive, class Key, class Data, class Compare, class Alloc >
	static void do_fix_save(Archive& ar, const std::multimap< Key, Data, Compare, Alloc >& v) {
		save_map(ar, v);
	}

	// LOAD PATH
	// most generic reading algo can be used for vectors, deques, lists, sets
	template< class Archive, class cont_t >
	static void do_fix_load(Archive& ar, cont_t& v) {
		serialize_fix_data< Archive > fixar(ar);

		// read container size
		std::size_t sz;
		fixar >> sz;

		// fill container using insert iterator -- most generic approach
		std::insert_iterator< cont_t > ii(v, v.begin());
		for(std::size_t i = 0; i < sz; ++i) {
			fixar >> *ii++;
		}
	}

	// optimized version for vectors with preallocation
	template< class Archive, class T, class Alloc >
	static void do_fix_load(Archive& ar, std::vector< T, Alloc >& v) {
		typedef std::vector< T, Alloc > vector_t;
		serialize_fix_data< Archive > fixar(ar);

		// read container size
		std::size_t sz;
		fixar >> sz;
		v.resize(sz);

		// read values
		typename vector_t::value_type t;
		for(typename vector_t::iterator i = v.begin(), end = v.end(); i != end; ++i) {
			fixar >> *i;
		}
	}

	// specialization for std::map
	template< class Archive, class map_t >
	static void load_map(Archive& ar, map_t& v) {
		typedef typename map_t::key_type key_t;
		typedef typename map_t::mapped_type data_t;
		serialize_fix_data< Archive > fixar(ar);

		// read container size
		std::size_t sz;
		fixar >> sz;

		// read data
		key_t k; data_t d;
		for(std::size_t i = 0; i < sz; ++i) {
			// load key & value
			fixar >> k >> d;
			// insert to map
			v[k] = d;
		}
	}

	// specialization for std::map
	template< class Archive, class Key, class Data, class Compare, class Alloc >
	static void do_fix_load(Archive& ar, std::map< Key, Data, Compare, Alloc >& v) {
		load_map(ar, v);
	}
	// and std::multimap
	template< class Archive, class Key, class Data, class Compare, class Alloc >
	static void do_fix_load(Archive& ar, std::multimap< Key, Data, Compare, Alloc >& v) {
		load_map(ar, v);
	}
};

// apply fixer to different containers
template< class T, class Alloc, class next_fixer >
struct serialize_fix_applicable< std::vector< T, Alloc >, serialize_fix_cont< next_fixer > > {
	typedef boost::true_type on_save;
	typedef boost::true_type on_load;
};
template< class T, class Alloc, class next_fixer >
struct serialize_fix_applicable< std::deque< T, Alloc >, serialize_fix_cont< next_fixer > > {
	typedef boost::true_type on_save;
	typedef boost::true_type on_load;
};
template< class T, class Alloc, class next_fixer >
struct serialize_fix_applicable< std::list< T, Alloc >, serialize_fix_cont< next_fixer > > {
	typedef boost::true_type on_save;
	typedef boost::true_type on_load;
};
template< class T, class Compare, class Alloc, class next_fixer >
struct serialize_fix_applicable< std::set< T, Compare, Alloc >, serialize_fix_cont< next_fixer > > {
	typedef boost::true_type on_save;
	typedef boost::true_type on_load;
};
template< class Key, class Data, class Compare, class Alloc, class next_fixer >
struct serialize_fix_applicable< std::map< Key, Data, Compare, Alloc >, serialize_fix_cont< next_fixer > > {
	typedef boost::true_type on_save;
	typedef boost::true_type on_load;
};
template< class T, class Compare, class Alloc, class next_fixer >
struct serialize_fix_applicable< std::multiset< T, Compare, Alloc >, serialize_fix_cont< next_fixer > > {
	typedef boost::true_type on_save;
	typedef boost::true_type on_load;
};
template< class Key, class Data, class Compare, class Alloc, class next_fixer >
struct serialize_fix_applicable< std::multimap< Key, Data, Compare, Alloc >, serialize_fix_cont< next_fixer > > {
	typedef boost::true_type on_save;
	typedef boost::true_type on_load;
};

} // eof blue_sky namespace

#endif /* end of include guard: BS_SERIALIZE_FIXCONT_W1TRZ4NG */

