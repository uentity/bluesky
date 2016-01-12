/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_PLUGIN_COMMON_H
#define BS_PLUGIN_COMMON_H

//#if defined(_WIN32) && defined(_MSC_VER)
	//#pragma warning(disable:4244)
//#endif

//#include "bs_python_kernel.h"
/*
#define BS_PY_TYPE_IMPL(T) \
	 type_descriptor T::bs_resolve_type() const { return sp->bs_resolve_type(); }

#define BS_PY_TYPE_DECL(T) \
	 public: type_descriptor bs_resolve_type() const;
*/


#define BSPY_OBJ_DECL_BEGIN(T)	\
	 class BS_API_PLUGIN T : public py_objbase {						\
	 public: T();

#define BSPY_OBJ_DECL_BEGIN_SHORT(T) \
	 BSPY_OBJ_DECL_BEGIN(py_ ## T)

#define BSPY_COM_DECL_BEGIN(T)					\
	 class BS_API_PLUGIN T : public py_objbase, public py_combase { \
	 public: T();

#define BSPY_COM_DECL_BEGIN_SHORT(T) \
	 BSPY_COM_DECL_BEGIN(py_ ## T)

#define BSPY_DECL_END };

#define BSPY_OBJ_DECL(T)					\
	 BSPY_OBJ_DECL_BEGIN_SHORT(T)	\
	 BSPY_DECL_END

#define BSPY_COM_DECL(T)					\
	 BSPY_COM_DECL_BEGIN_SHORT(T)	\
	 BSPY_DECL_END

#define BSPY_CLASSES_DECL(objclass,comclass)	\
	 BSPY_OBJ_DECL(objclass)					\
	 BSPY_COM_DECL(comclass)

	 /*BS_PY_TYPE_DECL(T)						\
T (); \
T (bs_py_object< type >::sp_type &); \
bs_objinst_holder instances(); \
virtual ~T() {}*/
/*
virtual void bs_free_this() const; \
virtual void bs_register_this() const; \
virtual ~T() {}*/

#define BSPY_OBJ_IMPL(T,wraps)								\
	 T::T() : py_objbase(give_kernel::Instance().create_object(wraps::bs_type())) {}

#define BSPY_OBJ_IMPL_SHORT(T)	\
	 BSPY_OBJ_IMPL(py_ ## T,T)

#define BSPY_COM_IMPL(T,wraps)								\
	 T::T() : py_objbase(give_kernel::Instance().create_object(wraps::bs_type())), \
			py_combase(smart_ptr<wraps>(sp)) {}

#define BSPY_COM_IMPL_SHORT(T)	\
	 BSPY_COM_IMPL(py_ ## T,T)

#define BSPY_CLASSES_IMPL(objclass,objwraps,comclass,comwraps)	\
	 BSPY_OBJ_IMPL(objclass,objwraps)					\
	 BSPY_COM_IMPL(comclass,comwraps)

#define BSPY_CLASSES_IMPL_SHORT(objclass,comclass)	\
	 BSPY_OBJ_IMPL_SHORT(objclass)					\
	 BSPY_COM_IMPL_SHORT(comclass)

#define BLUE_SKY_BOOST_PYTHON_MODULE_BEGIN(module_name) \
	 BOOST_PYTHON_MODULE(module_name) {

#define BLUE_SKY_BOOST_PYTHON_MODULE_END }

#define BS_EXPORT_OBJBASE_CLASS(name,python_name)	\
	 class_<name, bases <py_objbase> >(python_name)

#define BS_EXPORT_OBJBASE_CLASS_SHORT(name,python_name)	\
	 BS_EXPORT_OBJBASE_CLASS(py_ ## name,python_name)

#define BS_EXPORT_COMBASE_CLASS(name,python_name)	\
	 class_<name, bases <py_objbase, py_combase> >(python_name);

#define BS_EXPORT_COMBASE_CLASS_SHORT(name,python_name)	\
	 BS_EXPORT_COMBASE_CLASS(py_ ## name,python_name)

#define BS_DEF_EXPORT(classname,defname,python_defname) \
	 .def(python_defname,&classname::defname)

#define BS_DEF_EXPORT_POLICY(classname,defname,python_defname,manage_policy) \
	 .def(python_defname,&classname::defname,manage_policy())

#define BS_DEF_OPERATOR(op) \
	 .def(self op self)

#define BS_DEF_OPERATOR_RIGHT(op,l)							\
	 .def(self op l)

#define BS_DEF_EXPORT_SHORT(classname,defname)	\
	 BS_DEF_EXPORT(classname,defname, #defname)

#define BS_DEF_EXPORT_SHORT2(classname,defname)	\
	 BS_DEF_EXPORT_SHORT(py_ ## classname,defname)

#define BS_DEF_EXPORT_POLICY_SHORT(classname,defname,manage_policy)	\
	 BS_DEF_EXPORT_POLICY(classname,defname, #defname,manage_policy)

#define BS_DEF_EXPORT_POLICY_SHORT2(classname,defname,manage_policy)	\
	 BS_DEF_EXPORT_POLICY_SHORT(py_ ## classname,defname,manage_policy)

	 /*BS_PY_TYPE_IMPL(T)																									\
 T::T (bs_py_object< type >::sp_type &dest) : bs_py_object< type >(dest) {} \
 bs_objinst_holder T::instances() { \
		bs_objinst_holder oih; \
		copy(sp->bs_inst_begin(), sp->bs_inst_end(), oih.begin()); \
		return oih; } */
/* \
 void T::bs_register_this() const { sp.lock()->bs_register_instance(sp); }	\
 void T::bs_free_this() const { sp.lock()->bs_free_instance(sp); }*/

/*#define BS_PY_FACTORY_REG(T)									\
	 void register_obj()	{	\
		T::py_factory_index = factory_pair<T>::cntr; \
		obj_types().append(factory_pair<smart_ptr< T, true > >(T::bs_type(),create_py_ ## T));	\
	 }

#define BS_PY_CREATE_REG(T,based) \
	BS_API_PLUGIN smart_ptr<T,true> *create_py_ ## T (const smart_ptr<based, true> &sp) { \
		 return new smart_ptr<T,true>(sp); }*/

/*
		obj_types().append(factory_pair<T>(T::bs_type(),T::bs_create_copy)); \
	 }

#define BS_PY_CREATE_REG(T,based) \
	 BS_API_PLUGIN smart_ptr<T,true> *create_py_ ## T (smart_ptr<based, true> &sp) { \
			return new smart_ptr<T,true>(sp); }
*/
	 /*	BS_API_PLUGIN smart_ptr<T,true> *create_py_ ## T (smart_ptr<based, true> &sp) { \
			return new smart_ptr<T,true>(sp); }*/



#endif // BS_PLUGIN_COMMON_H
