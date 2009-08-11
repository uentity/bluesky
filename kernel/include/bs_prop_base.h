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
#include "bs_array.h"
#include "bs_map.h"

//! \defgroup tables tables - tables of blue-sky

namespace blue_sky {

//! type of table_t's key
#define DT_KEY_T typename table_t< T, cont_traits >::key_type
//! type of table_t's value
#define DT_VAL_T typename table_t< T, cont_traits >::value_type
//! type of table_t's reference to valued object
#define DT_REF_T typename table_t< T, cont_traits >::reference
#define DT_CONST_REF_T typename table_t< T, cont_traits >::const_reference
//! type of table_t's iterator
#define DT_ITER_T typename table_t< T, cont_traits >::iterator
//! type of table_t's const iterator
#define DT_CONST_ITER_T typename table_t< T, cont_traits >::const_iterator
#define DT_SP_TBL smart_ptr< table_t< T, cont_traits >, true >

/*!
	\class data_table
	\ingroup tables
	\brief container for holding different types of properties
*/
template< template< class, template< class > class > class table_t = bs_map, template< class > class cont_traits = str_val_traits >
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
	smart_ptr< table_t< T, cont_traits >, true > table() {
		typedef table_t< T, cont_traits > table_type;
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
	bool insert(const DT_KEY_T key, const DT_VAL_T& value) {
		return table< T >().lock()->insert(key, value);
	}

/*!
	\brief Add item method.
	\param value - value object
	\return such as std::type_traits
 */
	template< class T >
	bool insert(const DT_VAL_T& value) {
		return table< T >().lock()->insert(value);
	}

/*!
	\brief Remove item method.
	\param key - value object
*/
	template< class T>
	void erase(const DT_KEY_T& key) {
		table< T >().lock()->erase(key);
	}

	/*!
		\brief at() is a safe version of ss(). It operates only on existent tables and involves bounds-checking
		\return reference to value-object
	*/
template< class T >
#ifndef BS_DISABLE_MT_LOCKS
	bs_locker< T > at(const DT_KEY_T& key) {
		DT_SP_TBL p_vt = find_table< T >();
		if(!p_vt) throw std::out_of_range("str_val_table: Table of values of requested type doesn't exist");
		//NOTE: manual const_cast - I know what am I doing, but never do it by yourself unless you also know The Thing
		return bs_locker< T >(
			&const_cast< table_t< T, cont_traits >* >(p_vt.get())->at(key),
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
		table_t< T, cont_traits >::value_type()
#else
		typename table_t< T, cont_traits >::value_type()
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
			&const_cast< table_t< T, cont_traits >* >(p_vt.get())->operator[](key),
			*p_vt.mutex()
		);
#else
		return p_vt->operator[](key);
#endif
	}

	/*!
		\brief SubScripting operator.
		Forwards call to underlying table's operator[].
		\return const reference to value-object
	*/
	template< class T >
	DT_CONST_REF_T ss(const DT_KEY_T& key) const {
		DT_SP_TBL p_vt = find_table< T >();
		if(!p_vt) throw std::out_of_range("data_table: Table of values of requested type doesn't exist");
		return p_vt->operator[](key);
	}

	//! This is functional version of subscripting operator. Does assignment inside.
	template< class T >
	void set_item(const DT_KEY_T& key, const DT_VAL_T& value) {
		table< T >()->operator[](key) = value;
	}

	/*!
		\brief Check if table of given type already has been created
		\return true if table_t<T> created
	 */
	template< class T >
	bool is_table_created() const {
		typedef table_t< T, cont_traits > table_type;
		container::const_iterator tbl = tables_.find( BS_GET_TI(table_type) );
		return (tbl != tables_.end());
	};

	/*!
		\brief Search for created table of specified type.
		\return pointer to table_t<T>. If no such table found, returns NULL.
	*/
	template< class T >
	const smart_ptr< table_t< T, cont_traits >, true > find_table() const {
		typedef table_t< T, cont_traits > table_type;
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

	BLUE_SKY_TYPE_DECL_T(data_table);
};

//! type of map storage
typedef data_table< bs_map, str_val_traits > str_data_table;
//! type of vector storage
typedef data_table< bs_array, vector_traits > idx_data_table;

};	//namespace blue_sky

#endif
