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
  \file bs_common.cpp
  \brief Common parts of blue-sky
  \author Gagarin Alexander <gagrinav@ufanipi.ru>
*/
#include "bs_common.h"
#include "bs_kernel.h"
#include "type_descriptor.h"

#include <stdio.h>
//#include <iostream>

//using namespace blue_sky;
using namespace std;

#define BS_PLUGIN_DVERSION 0 //!< default plugin_descriptor version
#define BS_PLUGIN_IVERSION 0 //!< default plugin_initializer version
#define BS_TYPE_DVERSION 0 //!< default type_descriptor version
#define BS_NIL_PLUGIN_TAG "__blue_sky_nil_plugin__"
#define BS_NIL_TYPE_TAG "__blue_sky_nil_type__"

namespace blue_sky {

	//type() implementation for named_type
	//BS_TYPE_IMPL(named_type);

	plugin_initializer::plugin_initializer()
		: k_(give_kernel::Instance()) //kernel_version_(ver)
	{}

	unsigned int plugin_initializer::self_version() {
		return BS_PLUGIN_IVERSION;
	}

//------------------------------plugin_descriptor-----------------------------------------------------------
	plugin_descriptor::plugin_descriptor()
		 : name_(BS_NIL_PLUGIN_TAG), tag_()
	{}

	plugin_descriptor::plugin_descriptor(const char* name)
		 : name_(name), tag_()
	{}

	plugin_descriptor::plugin_descriptor(const BS_TYPE_INFO& plugin_tag, const char* name, const char* version,
		const char* short_descr, const char* long_descr, const char* py_namespace)
		 : name_(name), version_(version), short_descr_(short_descr), long_descr_(long_descr),
		 py_namespace_(py_namespace), tag_(plugin_tag)
	{}

	unsigned int plugin_descriptor::self_version() {
		return BS_PLUGIN_DVERSION;
	}

	bool plugin_descriptor::is_nil() const {
		return tag_.is_nil();
		//return (name_ == BS_NIL_PLUGIN_TAG);
	}

	bool plugin_descriptor::operator <(const plugin_descriptor& pd) const {
		return tag_ < pd.tag_;
	}

	bool plugin_descriptor::operator ==(const plugin_descriptor& pd) const {
		return tag_ == pd.tag_;
	}

	//plugin_descriptor::operator bool() {
	//	return (status_ == 0);
	//}

//------------------------------type_descriptor-------------------------------------------------------------

	//void type_descriptor::init(const BS_TYPE_OBJ& tp, const BS_OBJECT_CREATION_FUN cr_fn, const BS_OBJECT_COPY_FUN cp_fn,
	//	const std::string& stype, const std::string& short_descr, const std::string& long_descr)
	//{
	//	const_cast< BS_TYPE_OBJ& >(type_) = tp;
	//	const_cast< std::string& >(stype_) = stype;
	//	const_cast< std::string& >(short_descr_) = short_descr;
	//	const_cast< std::string& >(long_descr_) = long_descr;
	//	const_cast< BS_OBJECT_CREATION_FUN& >(creation_fun_) = cr_fn;
	//	const_cast< BS_OBJECT_COPY_FUN& >(copy_fun_) = cp_fn;
	//}

	type_descriptor::type_descriptor()
		: bs_ti_(), creation_fun_(NULL), copy_fun_(NULL), parent_td_fun_(NULL), stype_(BS_NIL_TYPE_TAG) //instances_(NULL)
	{}

	type_descriptor::type_descriptor(const BS_TYPE_INFO& tp, const BS_TYPE_CREATION_FUN& cr_fn, const BS_TYPE_COPY_FUN& cp_fn,
		const BS_GET_TD_FUN& parent_td_fn, const std::string& stype, const std::string& short_descr, const std::string& long_descr)
		: bs_ti_(tp), creation_fun_(cr_fn), copy_fun_(cp_fn), parent_td_fun_(parent_td_fn), //instances_(instances),
		stype_(stype), short_descr_(short_descr), long_descr_(long_descr)
	{
		//cout << "full type_descriptor constructered at " << this << " for type " << stype_ << endl;
	}

	type_descriptor::type_descriptor(const std::string& stype)
		: bs_ti_(), creation_fun_(NULL), copy_fun_(NULL), parent_td_fun_(NULL), stype_(stype)
	{}

	unsigned int type_descriptor::self_version()
	{
		return BS_TYPE_DVERSION;
	}

	type_descriptor type_descriptor::parent_td() const {
		if(parent_td_fun_)
			return (*parent_td_fun_)();
		else
			return type_descriptor();
	}

	//DEBUG
//	class test_nil {};
//	blue_sky::objbase* create_nil(bs_type_ctor_param) {
//		return static_cast< objbase* >((void*)(new test_nil()));
//	}
//	blue_sky::objbase* copy_nil(bs_type_cpy_ctor_param) {
//		return static_cast< objbase* >((void*)(new test_nil()));
//	}
//	type_descriptor nil_td() {
//		return type_descriptor();
//	}
//
//	type_descriptor td(Loki::Type2Type< test_nil >(), Loki::Type2Type< test_nil >(), Loki::Int2Type< 0 >(), "", "", "");

// 	type_descriptor::~type_descriptor() {
// 		cout << "type_descriptor at " << this;
// 		cout << " destructor called for type " << stype_ << endl;
// 	}

	//type_descriptor& type_descriptor::operator =(const type_descriptor& tp) {
	//	self_version_ = tp.self_version_;
	//	init(tp.type_, tp.creation_fun_, tp.copy_fun_, tp.stype_, tp.short_descr_, tp.long_descr_);
	//	return *this;
	//}
}
