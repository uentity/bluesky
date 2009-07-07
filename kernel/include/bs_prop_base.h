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

/*!
 * \file bs_prop_base.h
 * \brief Contains blue-sky storage tables.
 * \author uentity
 */
#ifndef _BS_PROP_BASE_H
#define _BS_PROP_BASE_H

#include "bs_kernel.h"

#include <map>
#ifdef UNIX
#include <ext/hash_map>
#else
#include <hash_map>
#endif

//! \defgroup tables tables - tables of blue-sky

namespace blue_sky {

	 /*!
		 \brief Blue-sky map.
		 \param key_type = type of map's key (first)
		 \param value_type = type of map's values (second)
	 */
	#ifdef UNIX
		#define BS_MAP(key_type, value_type) std::map< key_type, value_type >
		//#define BS_MAP(key_type, value_type) __gnu_cxx::hash_map< key_type, value_type >
	#else
		#define BS_MAP(key_type, value_type) std::map< key_type, value_type >
		//#define BS_MAP(key_type, value_type) stdext::hash_map< key_type, value_type >
	#endif

#if 0
	enum {
		bs_dt_values,
		bs_dt_smart_ptrs,
		bs_dt_ptrs,
		bs_dt_autodetect
	};

	template< class T, int traits_t >
	struct dt_val_traits {
		typedef T value_type;
	};

	template< class T >
	struct dt_val_traits< T, bs_dt_ptrs > {
		typedef T* value_type;
	};

	template< class T >
	struct dt_val_traits< T, bs_dt_smart_ptrs > {
		typedef smart_ptr< T > value_type;
	};

	template< class T >
	struct dt_val_traits< T, bs_dt_autodetect > {
		typedef typename dt_val_traits< T, conversion< T, objbase >::exists_uc >::value_type value_type;
	};

#else

	template< class T, int > struct vt_helper { typedef T value_type; };
	template< class T > struct vt_helper< T, 1 > { typedef smart_ptr< T, true > value_type; };

	template< class T >
	struct dt_val_traits {
	private:
		enum { objbase_child = blue_sky::conversion< T, objbase >::exists_uc };

	public:
		typedef typename vt_helper< T, objbase_child >::value_type value_type;
	};

#endif
	 /*!
		 class str_val_table<T>
		 \ingroup tables
		 \brief Implements an rray of key-value pairs addressed by string key
		*/
	template< class T >
	class str_val_table : public BS_MAP(std::string, typename dt_val_traits< T >::value_type), public objbase
	{
	public:
		 //! type of value
		typedef typename dt_val_traits< T >::value_type value_type;
		//typedef T value_type;
		//! type of key
		typedef std::string key_type;
		//! type of parent class
		typedef BS_MAP(key_type, value_type) container;
		//! type of map's pair
		typedef typename container::value_type data_pair;
		//! type of reference to second object (value)
		typedef typename container::mapped_type& reference;
		typedef const typename container::mapped_type& const_reference;
		//! type of map's iterator
		typedef typename container::iterator iterator;
		//! type of map's const iterator
		typedef typename container::const_iterator const_iterator;

		using container::erase;
		using container::find;
		using container::begin;
		using container::end;

		//!	Empty destructor.
		virtual ~str_val_table() {};

		/*!
			\brief Add item method.
			\param key - key object
			\param value - value object
			\return such as std::map
		 */
		bool add_item(const key_type& key, const value_type& value)	{
			return insert(data_pair(key, value)).second;
		}
		/*!
			\brief Remove item method.
			\param key - key object
		 */
		void rem_item(const key_type& key) {
			erase(key);
		}

		/*!
			\brief Search for item method.
			\param key - key object
		 */
		reference at(const key_type& key) {
			return _at< iterator, reference >(key);
		}

		const_reference at(const key_type& key) const {
			return const_cast< this_t* >(this)->_at< const_iterator, const_reference >(key);
		}

	private:
		typedef str_val_table< T > this_t;

		template< class iterator_t, class ref_t >
		ref_t _at(const key_type& key) {
			iterator_t p_res(find(key));
			if(p_res != end())
				return p_res->second;
			else
				throw std::out_of_range(std::string("str_val_table: no element found with key = ") + key);
		}

		BLUE_SKY_TYPE_STD_CREATE_T_MEM(str_val_table)
		BLUE_SKY_TYPE_STD_COPY_T_MEM(str_val_table)

		BLUE_SKY_TYPE_DECL_T_MEM(str_val_table, objbase, "str_val_table",
			"Array of values of the same type indexed by string key", "")
	};

	//default ctor implementation
	template< class T >
	str_val_table< T >::str_val_table(bs_type_ctor_param param)
		: objbase(param)
	{}

	//copy ctor implementation
	template< class T >
	str_val_table< T >::str_val_table(const str_val_table< T >& src) : bs_refcounter (src),
		 container(src), objbase(src)
	{}

	//creation and copy functions definitions
	//BLUE_SKY_TYPE_STD_CREATE_T_DEF(str_val_table< T, val_traits >, class T)
	//BLUE_SKY_TYPE_STD_COPY_T_DEF(str_val_table< T >, class T)

	//template definitions for common functions
	//BLUE_SKY_TYPE_IMPL_T_DEF(str_val_table< T >, objbase, class T)

	/*!
		class bs_array<T>
		\ingroup tables
		\brief Contains array of key-value pairs with integral key (index).

		std::vector< T > wrapper.
	 */
	template< class T >
	class bs_array : public objbase, public std::vector< T >
	{
	public:
		//! type of value
		//typedef T value_type;
		typedef typename dt_val_traits< T >::value_type value_type;
		//! type of vector of values
		typedef std::vector< value_type > container;
		//! type of key - index
		typedef typename container::size_type key_type;
		//! type of iterator
		typedef typename container::iterator iterator;
		//! type of const iterator
        typedef typename container::const_iterator const_iterator;
		//references
        typedef typename container::reference reference;
        typedef typename container::const_reference const_reference;

		using container::erase;
		using container::insert;
		using container::push_back;
		using container::size;
		using container::begin;
		using container::end;

		//! empty destructor
		virtual ~bs_array() {};

		/*!
			\brief Add item method.

			Insert after key-index object.
			\param key - key object
			\param value - value object
			\return such as std::vector
		 */
		bool add_item(const key_type& key, const value_type& value)	{
			if(key > size()) return false;
			insert(begin() + key, value);
			return true;
		}

		/*!
			\brief Add item method.

			Push back insert method.
			\param value - value object
			\return such as std::map
		 */
		bool add_item(const value_type& value)	{
			push_back(value);
			return true;
		}

		/*!
			\brief Remove item method.
			\param key - key object
		 */
		void rem_item(const key_type& key)	{
			erase(begin() + key);
		}

		//creation and copy functions definitions
		BLUE_SKY_TYPE_STD_CREATE_T_MEM(bs_array)
		BLUE_SKY_TYPE_STD_COPY_T_MEM(bs_array)

		BLUE_SKY_TYPE_DECL_T_MEM(bs_array, objbase, "bs_array",
			"Array of values of the same type indexed by integral type", "")
	};

	//default ctor implementation
	template< class T >
	bs_array< T >::bs_array(bs_type_ctor_param param)
		: bs_refcounter(), objbase(param)
	{}

	//copy ctor implementation
	template< class T >
	bs_array< T >::bs_array(const bs_array< T >& src)
		: bs_refcounter(), objbase(src), container(src)
	{}

	//creation and copy functions definitions
	//BLUE_SKY_TYPE_STD_CREATE_T_DEF(bs_array< T >, class T)
	//BLUE_SKY_TYPE_STD_COPY_T_DEF(bs_array< T >, class T)

	//template definitions for common functions
	//BLUE_SKY_TYPE_IMPL_T_DEF(bs_array< T >, objbase, class T)

	//! type of table_traits's key
	#define DT_KEY_T typename table_traits< T >::key_type
	//! type of table_traits's value
	#define DT_VAL_T typename table_traits< T >::value_type
	//! type of table_traits's reference to valued object
	#define DT_REF_T typename table_traits< T >::reference
	#define DT_CONST_REF_T typename table_traits< T >::const_reference
	//! type of table_traits's iterator
	#define DT_ITER_T typename table_traits< T >::iterator
	//! type of table_traits's const iterator
	#define DT_CONST_ITER_T typename table_traits< T >::const_iterator
	#define DT_SP_TBL smart_ptr< table_traits< T >, true >

	/*!
		\class data_table
		\ingroup tables
		\brief container for holding different types of properties
	*/
	template< template< class > class table_traits = str_val_table >
//		int val_traits = bs_dt_autodetect >
	class BS_API data_table : public objbase
	{
		friend class kernel;

		//! \brief creation function for kernel
		//static objbase* create_instance();

	public:
	//! container type - blue-sky map of BS_TYPE_OBJ and void*
		typedef BS_MAP(BS_TYPE_INFO, sp_obj) container;
	//! type of iterator
		typedef container::iterator iterator;
	//! type of const iterator
		typedef container::const_iterator const_iterator;

		//! empty destructor
		~data_table() {};

		//! creates or returns props table on demand
		template< class T >
		smart_ptr< table_traits< T >, true > table() {
			typedef table_traits< T > table_type;
			//static table_creater< T > tbl_(tables_);
			std::pair< container::iterator, bool > p_tbl(tables_.find(BS_GET_TI(table_type)), true);
			if(p_tbl.first == tables_.end())
				//insert new table
				p_tbl = tables_.insert( container::value_type(BS_GET_TI(table_type),
				give_kernel::Instance().create_object(table_type::bs_type())) );
			assert(p_tbl.first->second);
			return smart_ptr< table_type >(p_tbl.first->second, bs_dynamic_cast());
		}

		/*!
			\brief Add item method.
			\param key - key object
			\param value - value object
			\return such as std::type_traits
		 */
		template< class T >
		inline bool add_item(const DT_KEY_T key, const DT_VAL_T& value) {
			return table< T >().lock()->add_item(key, value);
		}

	/*!
		\brief Add item method.
		\param value - value object
		\return such as std::type_traits
	 */
		template< class T >
		inline bool add_item(const DT_VAL_T& value) {
			return table< T >().lock()->add_item(value);
		}

	/*!
		\brief Remove item method.
		\param key - value object
	*/
		template< class T>
		inline void rem_item(const DT_KEY_T& key) {
			table< T >().lock()->rem_item(key);
		}

		/*!
			\brief at() is a safe version of ss(). It operates only on existent tables and involves bounds-checking as it does
			std::vector::at().
			\return reference to value-object
		*/
template< class T >
#ifndef BS_DISABLE_MT_LOCKS
		bs_locker< T > at(const DT_KEY_T& key) {
			DT_SP_TBL p_vt = find_table< T >();
			if(!p_vt) throw std::out_of_range("str_val_table: Table of values of requested type doesn't exist");
			//NOTE: manual const_cast - I know what am I doing, but never do it by yourself unless you also know The Thing
			return bs_locker< T >(
				&const_cast< table_traits< T >* >(p_vt.get())->at(key),
				*p_vt.mutex()
				);
		};
#else
		DT_REF_T at(const DT_KEY_T& key) {
			DT_SP_TBL p_vt = find_table< T >();
			if(!p_vt) throw std::out_of_range("str_val_table: Table of values of requested type doesn't exist");
			return p_vt->at(key);
		};
#endif
		template< class T >
		DT_CONST_REF_T at(const DT_KEY_T& key) const {
			DT_SP_TBL p_vt = find_table< T >();
			if(!p_vt) throw std::out_of_range("str_val_table: Table of values of requested type doesn't exist");
			return p_vt->at(key);
		};

		/*!
		\brief Safely extract and return some value by given key. If no such value found,
		default value is returned.
		*/
		template< class T >
		DT_VAL_T extract_value(const DT_KEY_T& key, DT_CONST_REF_T def_value =
#ifdef _MSC_VER
			table_traits< T >::value_type()
#else
			typename table_traits< T >::value_type()
#endif
			) const {
			//try to find a value
			try {
				return at< T >(key);
			}
			catch(...) {
				return def_value;
			}
		}

		/*!
			\brief SubScripting operator.
			Forwards call to underlying table's operator[]. Creates table of given type if it isn't exists.
			\return reference to value-object
		*/
		//template< class T >
		//DT_CONST_REF_T ss(const DT_KEY_T& key) {
		//	return table< T >()->operator[](key);
		//}
		template< class T >
#ifndef BS_DISABLE_MT_LOCKS
		bs_locker< T >
#else
		DT_REF_T
#endif
		ss(const DT_KEY_T& key) {
			DT_SP_TBL p_vt = table< T >();
#ifndef BS_DISABLE_MT_LOCKS
			//NOTE: manual const_cast - I know what am I doing, but never do it by yourself unless you also know The Thing
			return bs_locker< T >(
				&const_cast< table_traits< T >* >(p_vt.get())->operator[](key),
				*p_vt.mutex()
			);
#else
			return p_vt->operator[](key);
#endif
		}

		//! This is functional version of subscripting operator. Does assignment inside.
		template< class T >
		void set_item(const DT_KEY_T& key, const DT_VAL_T& value) {
			table< T >()->operator[](key) = value;
		}

		/*!
			\brief Check if table of given type already has been created
			\return true if table_traits<T> created
		 */
		template< class T >
		bool is_table_created() const {
			typedef table_traits< T > table_type;
			container::const_iterator tbl = tables_.find( BS_GET_TI(table_type) );
			return (tbl != tables_.end());
		};

		/*!
			\brief Search for created table of specified type.
			\return pointer to table_traits<T>. If no such table found, returns NULL.
		*/
		template< class T >
		const smart_ptr< table_traits< T >, true > find_table() const {
			typedef table_traits< T > table_type;
			container::const_iterator tbl = tables_.find( BS_GET_TI(table_type) );
			if(tbl != tables_.end())
				return smart_ptr< table_type, true >(tbl->second);
			else return NULL;
		}

		/*!
			\brief Table size quering method
			\return size of table
		*/
		template< class T >
		inline ulong size() const {
			DT_SP_TBL p_vt = find_table< T >();
			if(!p_vt) return 0;
			else p_vt->size();
		}

		//begin() and end() to iterate through props tables in stl-style

		//! begin() to iterate through props tables in stl-style - const version
		template< class T >
		inline DT_CONST_ITER_T begin() const {
			return table< T >()->begin();
		}

		//! end() to iterate through props tables in stl-style - const version
		template< class T >
		inline DT_CONST_ITER_T end() const {
			return table< T >()->end();
		}


	private:
		container tables_; //!< object container
		//! standard constructor
		//data_table() {};
		// constructor with parameter
		//data_table(sp_obj param);

		//! \brief wrapper class that registers created props table in tables_
		//template< class T >
		//struct table_creater
		//{
		//	typedef table_traits< T > table_type; //!< type of table
		//	table_type table_;

		//	/*!
		//	\brief Default constructor.

		//	Inserts internal table table_ into h-container.
		//	\param h - BS_MAP(BS_TYPE_OBJ, void*)
		//	*/
		//	explicit table_creater(container& h)
		//	{
		//		h[ BS_TYPE_INFO(typeid(table_type)) ] = table_;
		//	}

		//	/*!
		//	\brief Execute operator of table_creator.
		//	\return reference to internal table
		//	*/
		//	table_type& operator()() { return table_; }
		//};

		BLUE_SKY_TYPE_DECL_T(data_table);
	};

	//! type of map storage
	typedef data_table< str_val_table > str_data_table;
	//! type of vector storage
	typedef data_table< bs_array > idx_data_table;

//	template< class T, template< class > class table_traits >
//	bool dt_extract_value(const sp_obj p, const DT_KEY_T& key, DT_REF_T res, DT_CONST_REF_T def_value =
//#ifdef _MSC_VER
//		table_traits< T >::value_type()
//#else
//		DT_VAL_T
//#endif
//	)
//	{
//		//check if p_dt really points to data_table
//		smart_ptr< data_table< table_traits > > p_dt(p, bs_dynamic_cast());
//		if(!p_dt) {
//			res = def_value;
//			return false;
//		}
//		else
//			return p_dt->extract_value< T >(key, res, def_value);
//	}

};	//namespace blue_sky

#endif
