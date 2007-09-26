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

//needed to build old-fashioned plugins
#ifdef _LIBAPI__
	#undef _LIBAPI__
#endif
#ifdef _CLASS_DECLSPEC__
	#undef _CLASS_DECLSPEC__
#endif
#ifdef _LIBAPI_PLUGIN_
	#undef _LIBAPI_PLUGIN_
#endif
#ifdef _CLASS_DECLSPEC_PLUGIN_
	#undef _CLASS_DECLSPEC_PLUGIN_
#endif

#define _LIBAPI__ BS_C_API
#define _CLASS_DECLSPEC__ BS_API
#define _LIBAPI_PLUGIN_ BS_C_API_PLUGIN
#define _CLASS_DECLSPEC_PLUGIN_ BS_API_PLUGIN

#define BS_COM_REG_INSTANCE_IMPL(T, obj, base) \
	void T::bs_register_instance(const smart_ptr< T, true >& sp_inst) { \
	base::bs_register_instance(sp_inst); \
	if(find(inst_list_.begin(), inst_list_.end(), sp_inst) == inst_list_.end()) \
	inst_list_.push_back(sp_inst); \
	if(find(obj::bs_commands().begin(), obj::bs_commands().end(), sp_inst) == obj::bs_commands().end()) \
	obj::bs_commands().push_back(sp_inst); \
}

#define BS_COM_REG_INSTANCE_IMPL_T(T, obj, base) \
	template< > BS_API_PLUGIN \
	BS_COM_REG_INSTANCE_IMPL(T, obj, base)

#define BS_COM_FREE_INSTANCE_IMPL(T, obj, base) \
	void T::bs_free_instance(const smart_ptr< T, true >& sp_inst) { \
	inst_list_.remove(sp_inst); \
	obj::bs_commands().remove(sp_inst); \
	base::bs_free_instance(sp_inst); \
}

#define BS_COM_FREE_INSTANCE_IMPL_T(T, obj, base) \
	template< > BS_API_PLUGIN \
	BS_COM_FREE_INSTANCE_IMPL(T, obj, base)

#define BLUE_SKY_COM_DECL(T) \
	BS_COMMON_DECL(T) \
	BS_TYPE_DECL \
	BS_LOCK_THIS_DECL(T)

#define BLUE_SKY_COM_DECL_T(T) \
	BLUE_SKY_COM_DECL(T)

#define BLUE_SKY_COM_IMPL(T, obj, base, type_string, short_descr, long_descr) \
	BS_COM_REG_INSTANCE_IMPL(T, obj, base) \
	BS_COM_FREE_INSTANCE_IMPL(T, obj, base) \
	BS_COMMON_IMPL(T) \
	BS_TYPE_IMPL(T, type_string, short_descr, long_descr)

#define BLUE_SKY_COM_IMPL_SHORT(T, obj, base, short_descr) \
	BLUE_SKY_COM_IMPL(T, obj, base, #T, short_descr, "")

#define BLUE_SKY_COM_IMPL_T(T, obj, base, short_descr, long_descr) \
	BS_COM_REG_INSTANCE_IMPL_T(T, obj, base) \
	BS_COM_FREE_INSTANCE_IMPL_T(T, obj, base) \
	BS_COMMON_IMPL_T(T) \
	BS_TYPE_IMPL_T(T, type_string, short_descr, long_descr) \
	template class T;

#define BLUE_SKY_COM_IMPL_T_SHORT(T, obj, base, short_descr) \
	BLUE_SKY_COM_IMPL_T(T, obj, base, #T, short_descr, "")

///*!
//\brief Free command instance method.
//\param sp_inst - smart pointer blue_sky::combase::sp_com, which necessary to free.
//*/
//void combase::bs_free_instance(const smart_ptr< combase, true >& sp_inst) {
//	instances_.lock()->remove(sp_inst);
//	objbase::bs_commands().remove(sp_inst);
//}
//
///*!
//\brief Register command instance method.
//\param sp_inst - smart pointer blue_sky::combase::sp_com, which necessary to register.
//*/
//void combase::bs_register_instance(const smart_ptr< combase, true >& sp_inst) {
//	{
//		lsmart_ptr< sp_objinst > l_inst(instances_);
//		if(std::find(l_inst->begin(), l_inst->end(), sp_inst) == l_inst->end())
//			l_inst->push_back(sp_inst);
//	}
//	if(std::find(objbase::bs_commands().begin(), objbase::bs_commands().end(), sp_inst) == objbase::bs_commands().end())
//		objbase::bs_commands().push_back(sp_inst);
//}
