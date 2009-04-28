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
  \file bs_object_base.h
  \brief declaration of base object class for neo project
  \author Gagarin Alexander <gagrinav@ufanipi.ru>
  \date 2007-03-04
 */
#ifndef _BS_OBJECT_BASE_H
#define _BS_OBJECT_BASE_H

#include "bs_common.h"
#include "bs_refcounter.h"
#include "type_descriptor.h"
#include "bs_messaging.h"

#include "boost/preprocessor/tuple/rem.hpp"

//! \defgroup object_base - base class of objects and commands in BlueSky

//! BlueSky kernel instance getter macro
#define BS_KERNEL blue_sky::give_kernel::Instance()


#if 0
/*!
\brief Function to clear all static lists that contains smart pointer to given instance of BlueSky type T
\param T - type of your derived class
\param base = type of base class from which T is inherited
This macro is included by BLUE_SKY_TYPE_IMPL*
*/

#define BS_TYPE_FREE_INSTANCE_BODY(T, base) \
bs_free_instance(const smart_ptr< T, true >& sp_inst) { \
	bs_instances().lock()->remove(sp_inst); \
	base::bs_free_instance(sp_inst); \
}
#define BS_TYPE_FREE_INSTANCE_IMPL(T, base) void T::BS_TYPE_FREE_INSTANCE_BODY(T, base)

/*!
\brief Function that registeres given instance of type T in all static lists of instances up the class hierarchy
\param T - type of your derived class
\param base = type of base class from which T is inherited
This macro is included by BLUE_SKY_TYPE_IMPL*
*/
#define BS_TYPE_REG_INSTANCE_BODY(T, base) \
bs_register_instance(const smart_ptr< T, true >& sp_inst) { \
	base::bs_register_instance(sp_inst); \
	lsmart_ptr< sp_objinst > l_inst(bs_instances()); \
	if(std::find(l_inst->begin(), l_inst->end(), sp_inst) == l_inst->end()) {\
		l_inst->push_back(sp_inst); return true; }\
	else return false; \
}
#define BS_TYPE_REG_INSTANCE_IMPL(T, base) bool T::BS_TYPE_REG_INSTANCE_BODY(T, base)

/*!
	\brief Implementation of insntance releasing function for templated class specialization
	Included by BLUE_SKY_TYPE_IMPL_T
 */
#define BS_TYPE_FREE_INSTANCE_IMPL_T(T, base) \
template< > BS_API_PLUGIN BS_TYPE_FREE_INSTANCE_IMPL(T, base) \
template< > BS_API_PLUGIN void T::bs_free_this() const { T::bs_free_instance(this); }

/*!
	\brief Implementation of instance registration function for templated type specialization
	Included by BLUE_SKY_TYPE_IMPL_T
*/
#define BS_TYPE_REG_INSTANCE_IMPL_T(T, base) \
template< > BS_API_PLUGIN BS_TYPE_REG_INSTANCE_IMPL(T, base) \
template< > BS_API_PLUGIN bool T::bs_register_this() const { return T::bs_register_instance(this); }
#endif //0

/*!
\brief Very common declarations for both objbase and command types/

*	Contains type identification support as well as static list of instances of this type and
*	functions to manipulate with it.
*	This macro is included by BLUE_SKY_TYPE_DECL*
*/
#define BS_COMMON_DECL(T) \
public: static bs_objinst_holder::const_iterator bs_inst_begin(); \
static bs_objinst_holder::const_iterator bs_inst_end(); \
static ulong bs_inst_cnt(); \
protected: T(bs_type_ctor_param param = NULL); \
T(const T &x);
/*static sp_objinst bs_instances(); \
static void bs_free_instance(const smart_ptr< T, true >& sp_inst); \
static bool bs_register_instance(const smart_ptr< T, true >& sp_inst); \
virtual void bs_free_this() const; \
virtual bool bs_register_this() const; */

#define BS_COMMON_DECL_MEM(T)																			\
public: static bs_objinst_holder::const_iterator bs_inst_begin()	\
	{																																\
		return BS_KERNEL.objinst_begin (bs_type ());									\
	}																																\
	static bs_objinst_holder::const_iterator bs_inst_end()					\
	{																																\
		return BS_KERNEL.objinst_end (bs_type ());										\
	}																																\
	static ulong bs_inst_cnt()																			\
	{																																\
		return BS_KERNEL.objinst_cnt(bs_type());											\
	}


/*!
\brief Very common implementations of functions for working with static list of instances for templated type.
*/
#define BS_COMMON_IMPL_EXT_(prefix, T, is_decl) \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::bs_objinst_holder::const_iterator \
BOOST_PP_SEQ_ENUM(BOOST_PP_IIF(is_decl, (), T))BOOST_PP_IIF(is_decl, BOOST_PP_EMPTY(), ::)\
bs_inst_begin() { return BS_KERNEL.objinst_begin(bs_type()); } \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::bs_objinst_holder::const_iterator \
BOOST_PP_SEQ_ENUM(BOOST_PP_IIF(is_decl, (), T))BOOST_PP_IIF(is_decl, BOOST_PP_EMPTY(), ::)\
bs_inst_end() { return BS_KERNEL.objinst_end(bs_type()); } \
BOOST_PP_SEQ_ENUM(prefix) ulong \
BOOST_PP_SEQ_ENUM(BOOST_PP_IIF(is_decl, (), T))BOOST_PP_IIF(is_decl, BOOST_PP_EMPTY(), ::)\
bs_inst_cnt() { return BS_KERNEL.objinst_cnt(bs_type()); }

#define BS_COMMON_DECL_T_MEM(T) \
BS_COMMON_IMPL_EXT_((public: static), (T), 1) \
protected: T(bs_type_ctor_param param = NULL); \
T(const T&); \
/*
public: static blue_sky::bs_objinst_holder::const_iterator bs_inst_begin() \
	{ return BS_KERNEL.objinst_begin(bs_type()); } \
static blue_sky::bs_objinst_holder::const_iterator bs_inst_end() \
	{ return BS_KERNEL.objinst_end(bs_type()); } \
static ulong bs_inst_cnt() { return BS_KERNEL.objinst_cnt(bs_type()); } \
*/

/*!
\brief Very common implementations of functions for working with static list of instances.
*	This macro is included by BLUE_SKY_TYPE_IMPL*
*/
#define BS_COMMON_IMPL(T) \
BS_COMMON_IMPL_EXT_((), (T), 0)
/*
blue_sky::bs_objinst_holder::const_iterator T::bs_inst_begin() { return BS_KERNEL.objinst_begin(bs_type()); } \
blue_sky::bs_objinst_holder::const_iterator T::bs_inst_end() { return BS_KERNEL.objinst_end(bs_type()); } \
ulong T::bs_inst_cnt() { return BS_KERNEL.objinst_cnt(bs_type()); }
*/

/*!
\brief Very common implementations of functions for working with static list of instances for templated type specialization.
*/
#define BS_COMMON_IMPL_T(T) \
BS_COMMON_IMPL_EXT_((template< > BS_API_PLUGIN), (T), 0)
/*
template< > BS_API_PLUGIN blue_sky::bs_objinst_holder::const_iterator T::bs_inst_begin() \
	{ return BS_KERNEL.objinst_begin(bs_type()); } \
template< > BS_API_PLUGIN blue_sky::bs_objinst_holder::const_iterator T::bs_inst_end() \
	{ return BS_KERNEL.objinst_end(bs_type()); } \
template< > BS_API_PLUGIN ulong T::bs_inst_cnt() { return BS_KERNEL.objinst_cnt(bs_type()); }
*/

//! put T definition into round braces
#define BS_COMMON_IMPL_T_EXT(t_params_num, T) \
BS_COMMON_IMPL_EXT_((template< > BS_API_PLUGIN), BOOST_PP_TUPLE_TO_SEQ(t_params_num, T), 0)

/*!
\brief Very common implementations of functions for working with static list of instances for templated types.
 Generates common template functions definitions for classes with arbitrary template parameters
*/
#define BS_COMMON_IMPL_T_DEF(T, t_params) BS_COMMON_IMPL_EXT_(\
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (template< BS_TLIST_FORMER(t_params) > BS_API_PLUGIN)), \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (T< BS_CLIST_FORMER(t_params) >)), 0)

//------------------------declarations---------------------------------------------
/*!
\brief This function needed to access non-constant members from constant member functions.
\return Proxy bs_locker object.
*/
#define BS_LOCK_THIS_DECL(T) \
public: lsmart_ptr< smart_ptr< T, true > > lock() const \
{ assert(this); return lsmart_ptr< smart_ptr< T, true > >(smart_ptr< T, true >(this)); }

/*!
	\brief Put this macro in the end of your BlueSky non-templated type declaration
	\param T = name of your class
 */
#define BLUE_SKY_TYPE_DECL(T) \
BS_TYPE_DECL \
BS_COMMON_DECL(T) \
BS_LOCK_THIS_DECL(T)

/*!
	\brief Put this macro in the end of your BlueSky templated type declaration
	\param T = name of your class
	To be used with templated types which export specializations
*/
#define BLUE_SKY_TYPE_DECL_T(T) \
BLUE_SKY_TYPE_DECL(T)

/*!
\brief Put this macro in the end of your BlueSky templated type declaration
\param T = name of your class
\param common_stype = common string type prefix that will be shared between all specializations of type T.
\param short_descr = shared short description of type
\param long_descr = shared long description of type
To be used with templated types which export source definition. Specializations are instantiated in client code.
When creating specialization of type T, client should pass unique string type postfix to macro BS_TYPE_IMPL_T_DEF.
Complete unique string type for given template specialization will be: common_stype + stype_postfix
*/
#define BLUE_SKY_TYPE_DECL_T_MEM(T, base, common_stype, short_descr, long_descr) \
BS_TYPE_DECL_T_MEM(T, base, common_stype, short_descr, long_descr) \
BS_COMMON_DECL_T_MEM(T) \
BS_LOCK_THIS_DECL(T)

//! same as BLUE_SKY_TYPE_DECL_T_MEM but not for template classes
#define BLUE_SKY_TYPE_DECL_MEM(T, base, common_stype, short_descr, long_descr)	\
	BS_TYPE_DECL_MEM((T), (base), common_stype, short_descr, long_descr, false)		\
	BS_COMMON_DECL_MEM(T)																													\
	BS_LOCK_THIS_DECL(T)

//------------------------ implementation ------------------------------------------------------------------------------
/*#define BS_TYPE_IMPL_COMMON(T, base) \
BS_COMMON_IMPL(T) \
BS_TYPE_REG_INSTANCE_IMPL(T, base) \
BS_TYPE_FREE_INSTANCE_IMPL(T, base) */

#define BLUE_SKY_TYPE_IMPL(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL(T, base, type_string, short_descr, long_descr) \
BS_COMMON_IMPL(T)

#define BLUE_SKY_TYPE_IMPL_NOCOPY(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_NOCOPY(T, base, type_string, short_descr, long_descr) \
BS_COMMON_IMPL(T)

#define BLUE_SKY_TYPE_IMPL_SHORT(T, base, short_descr) \
BLUE_SKY_TYPE_IMPL(T, base, #T, short_descr, "")

#define BLUE_SKY_TYPE_IMPL_NOCOPY_SHORT(T, base, short_descr) \
BLUE_SKY_TYPE_IMPL_NOCOPY(T, base, #T, short_descr, "")

//---------------- templated implementation - for partial specializations creation -------------------------------------
/*#define BS_TYPE_IMPL_T_COMMON(T, base) \
BS_COMMON_IMPL_T(T) \
BS_TYPE_REG_INSTANCE_IMPL_T(T, base) \
BS_TYPE_FREE_INSTANCE_IMPL_T(T, base) */

#define BLUE_SKY_TYPE_IMPL_T(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_T(T, base, type_string, short_descr, long_descr) \
BS_COMMON_IMPL_T(T) \
template class T;

#define BLUE_SKY_TYPE_IMPL_T_NOCOPY(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_T_NOCOPY(T, base, type_string, short_descr, long_descr) \
BS_COMMON_IMPL_T(T) \
template class T;

//! surround your class's and base's defintions with round braces
#define BLUE_SKY_TYPE_IMPL_T_EXT(T_tup_size, T_tup, base_tup_size, base_tup, type_string, short_descr, long_descr, nocopy) \
BS_TYPE_IMPL_T_EXT(T_tup_size, T_tup, base_tup_size, base_tup, type_string, short_descr, long_descr, nocopy) \
BS_COMMON_IMPL_T_EXT(T_tup_size, T_tup) \
template class BOOST_PP_TUPLE_REM_CTOR(T_tup_size, T_tup);

#define BLUE_SKY_TYPE_IMPL_T_SHORT(T, base, short_descr) \
BLUE_SKY_TYPE_IMPL_T(T, base, #T, short_descr, "")

#define BLUE_SKY_TYPE_IMPL_T_NOCOPY_SHORT(T, base, short_descr) \
BLUE_SKY_TYPE_IMPL_T_NOCOPY(T, base, #T, short_descr, "")

//============================================= End of Macro ===========================================================

namespace blue_sky {
/*!
 * \brief object_base ... short description ...
 * \author Gagarin Alexander <gagrinav@ufanipi.ru>
 * \date 2007-03-04
 * ... description ...
 */

/*!
	\class objbase
	\ingroup object_base
	\brief This is a base class for all objects.
*/

	class BS_API objbase : virtual public bs_refcounter, public bs_messaging
	{
		friend class kernel;
		friend class combase;
		friend class bs_inode;
		friend class bs_link;

	public:
		//typedef smart_ptr< objbase, true > sp_obj;

		//signals list
		BLUE_SKY_SIGNALS_DECL_BEGIN(bs_messaging)
			on_unlock,
		BLUE_SKY_SIGNALS_DECL_END

		//! type_descriptor of objbase class
		static type_descriptor bs_type();

		/*!
		\brief Type descriptor resolver for derived class.
		This method should be overridden by childs.
		\return reference to const type_descriptor
		*/
		virtual type_descriptor bs_resolve_type() const = 0;

		//register this instance in kernel instances list
		int bs_register_this() const;
		//remove this instance from kernel instances list
		int bs_free_this() const;

		//default object deletion method - executes 'delete this'
		virtual void dispose() const;

		//virtual destructor
		virtual ~objbase();

		/*!
		\brief Access to corresponding inode object
		*/
		smart_ptr< blue_sky::bs_inode, true > inode() const;

	protected:
		//! Constructor for derived classes - accepts signals range
		objbase(const bs_messaging::sig_range_t&);

		//associated inode
		smart_ptr< blue_sky::bs_inode, true > inode_;

		//! swap function needed to provide assignment ability
		void swap(objbase& rhs);

		//! assignamnt operator
		//objbase& operator =(const objbase& obj);

		//! Necessary declarations.
		BS_COMMON_DECL(objbase);
		BS_LOCK_THIS_DECL(objbase);
	};

}	//namespace blue_sky

#endif /* #ifndef _BS_OBJECT_BASE_H */
