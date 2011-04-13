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

/// @file bs_object_base_macro.h
/// @brief Main macro definitions to ease creating new BlueSky object
/// @author Alexander Gagarin aka uentity
/// @date 2009-06-26

#ifndef _BS_OBJECT_BASE_MACRO_H
#define _BS_OBJECT_BASE_MACRO_H

#include "boost/preprocessor/tuple/rem.hpp"

//! BlueSky kernel instance getter macro
#define BS_KERNEL blue_sky::give_kernel::Instance()

/*!
\brief Very common declarations for both objbase and command types

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

#define BS_COMMON_DECL_MEM(T)                                        \
    public: static bs_objinst_holder::const_iterator bs_inst_begin() \
    {                                                                \
        return BS_KERNEL.objinst_begin (bs_type ());                 \
    }                                                                \
    static bs_objinst_holder::const_iterator bs_inst_end()           \
    {                                                                \
        return BS_KERNEL.objinst_end (bs_type ());                   \
    }                                                                \
    static ulong bs_inst_cnt()                                       \
    {                                                                \
        return BS_KERNEL.objinst_cnt(bs_type());                     \
    }


/*!
\brief Very common implementations of functions for working with static list of instances for templated type.
*/
#define BS_COMMON_IMPL_EXT_(prefix, T, is_decl)                                                    \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::bs_objinst_holder::const_iterator BS_FMT_TYPE_SPEC(T, is_decl) \
bs_inst_begin() { return BS_KERNEL.objinst_begin(bs_type()); }                                     \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::bs_objinst_holder::const_iterator BS_FMT_TYPE_SPEC(T, is_decl) \
bs_inst_end() { return BS_KERNEL.objinst_end(bs_type()); }                                         \
BOOST_PP_SEQ_ENUM(prefix) ulong BS_FMT_TYPE_SPEC(T, is_decl)                                       \
bs_inst_cnt() { return BS_KERNEL.objinst_cnt(bs_type()); }

#define BS_COMMON_DECL_T_MEM(T)                \
BS_COMMON_IMPL_EXT_((public: static), (T), 1)  \
protected: T(bs_type_ctor_param param = NULL); \
T(const T&);

/*!
\brief Very common implementations of functions for working with static list of instances.
*	This macro is included by BLUE_SKY_TYPE_IMPL*
*/
#define BS_COMMON_IMPL(T) \
BS_COMMON_IMPL_EXT_(BS_SEQ_NIL(), (T), 0)

/*!
\brief Very common implementations of functions for working with static list of instances for templated type specialization.
*/
#define BS_COMMON_IMPL_T(T) \
BS_COMMON_IMPL_EXT_((template< > BS_API_PLUGIN), (T), 0)

//! put T definition into round braces
#define BS_COMMON_IMPL_T_EXT(t_params_num, T) \
BS_COMMON_IMPL_EXT_((template< > BS_API_PLUGIN), BOOST_PP_TUPLE_TO_SEQ(t_params_num, T), 0)

/*!
\brief Very common implementations of functions for working with static list of instances for templated types.
 Generates common template functions definitions for classes with arbitrary template parameters
*/
#define BS_COMMON_IMPL_T_DEF(T, t_params) BS_COMMON_IMPL_EXT_(                                             \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (template< BS_TLIST_FORMER(t_params) > BS_API_PLUGIN)), \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (T< BS_CLIST_FORMER(t_params) >)), 0)

//------------------------declarations---------------------------------------------
/*!
\brief This function needed to access non-constant members from constant member functions.
\return Proxy bs_locker object.
*/
#define BS_LOCK_THIS_DECL(T)                                                             \
public: lsmart_ptr< smart_ptr< T, true > > lock() const                                  \
{ assert(this); return lsmart_ptr< smart_ptr< T, true > >(smart_ptr< T, true >(this)); }

/*!
	\brief Put this macro in the end of your BlueSky non-templated type declaration
	\param T = name of your class
 */
#define BLUE_SKY_TYPE_DECL(T) \
BS_TYPE_DECL                  \
BS_COMMON_DECL(T)             \
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
BS_TYPE_DECL_T_MEM(T, base, common_stype, short_descr, long_descr)               \
BS_COMMON_DECL_T_MEM(T)                                                          \
BS_LOCK_THIS_DECL(T)

//! same as BLUE_SKY_TYPE_DECL_T_MEM but not for template classes
#define BLUE_SKY_TYPE_DECL_MEM(T, base, common_stype, short_descr, long_descr) \
BS_TYPE_DECL_MEM((T), (base), common_stype, short_descr, long_descr, false)    \
BS_COMMON_DECL_MEM(T)                                                          \
BS_LOCK_THIS_DECL(T)

//------------------------ implementation ------------------------------------------------------------------------------
#define BLUE_SKY_TYPE_IMPL(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL(T, base, type_string, short_descr, long_descr)               \
BS_COMMON_IMPL(T)

#define BLUE_SKY_TYPE_IMPL_NOCOPY(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_NOCOPY(T, base, type_string, short_descr, long_descr)               \
BS_COMMON_IMPL(T)

#define BLUE_SKY_TYPE_IMPL_SHORT(T, base, short_descr) \
BLUE_SKY_TYPE_IMPL(T, base, #T, short_descr, "")

#define BLUE_SKY_TYPE_IMPL_NOCOPY_SHORT(T, base, short_descr) \
BLUE_SKY_TYPE_IMPL_NOCOPY(T, base, #T, short_descr, "")

//---------------- templated implementation - for partial specializations creation -------------------------------------
#define BLUE_SKY_TYPE_IMPL_T(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_T(T, base, type_string, short_descr, long_descr)               \
BS_COMMON_IMPL_T(T)                                                         \
template class T;

#define BLUE_SKY_TYPE_IMPL_T_NOCOPY(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_T_NOCOPY(T, base, type_string, short_descr, long_descr)               \
BS_COMMON_IMPL_T(T)                                                                \
template class T;

//! surround your class's and base's defintions with round braces
#define BLUE_SKY_TYPE_IMPL_T_EXT(T_tup_size, T_tup, base_tup_size, base_tup, type_string, short_descr, long_descr, nocopy) \
BS_TYPE_IMPL_T_EXT(T_tup_size, T_tup, base_tup_size, base_tup, type_string, short_descr, long_descr, nocopy)               \
BS_COMMON_IMPL_T_EXT(T_tup_size, T_tup)                                                                                    \
template class BOOST_PP_TUPLE_REM_CTOR(T_tup_size, T_tup);

#define BLUE_SKY_TYPE_IMPL_T_SHORT(T, base, short_descr) \
BLUE_SKY_TYPE_IMPL_T(T, base, #T, short_descr, "")

#define BLUE_SKY_TYPE_IMPL_T_NOCOPY_SHORT(T, base, short_descr) \
BLUE_SKY_TYPE_IMPL_T_NOCOPY(T, base, #T, short_descr, "")

// shortcut for quick declaration of smart_ptr to BS object
#define BS_SP(T) smart_ptr< T, true >
// shortcut for quick declaration of smart_ptr to generic object
#define BS_SPG(T) smart_ptr< T, false >

#endif	// guard

