/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "py_bs_object_base.h"
#include "py_bs_exports.h"

#include <boost/python/enum.hpp>

namespace blue_sky {
namespace python {

py_objbase::py_objbase(const sp_obj &tsp)
	: py_bs_messaging(tsp),sp(tsp),inode_(NULL)//,inode_(sp->inode())
{
	if (sp)
		inode_ = py_bs_inode(sp->inode());
}

py_objbase::py_objbase(const type_descriptor& td)
	: py_bs_messaging(NULL), inode_(NULL)
{
	sp = BS_KERNEL.create_object(td);
	assert(sp);
	spmsg = sp;
	inode_.spinode = sp->inode();
}

py_objbase::~py_objbase() {}

type_descriptor py_objbase::bs_resolve_type() const {
	return sp->bs_resolve_type();
}

void py_objbase::ptr () {
	printf ("%p\n", sp.get());
}

sp_obj py_objbase::get_sp() const {
	return sp;
}

ulong py_objbase::refs() const {
	return sp.refs();
}
//		const sp_obj py_objbase::get_sp() const {
//			return sp;
//		}

py_bs_inode py_objbase::inode() const {
	return inode_;
}

bool py_objbase::fire_signal(int signal_code, const py_objbase& params) const {
	return sp->fire_signal(signal_code,params.sp);
}

std::vector< int > py_objbase::get_signal_list() const {
	return sp->get_signal_list();
}

//void feedback(int signal_code) {
//}

//void link(const python_slot&);

py_holder_iterator py_objbase::begin() {
	return py_holder_iterator(objbase::bs_inst_begin(),&(objbase::bs_inst_end));
}

py_holder_iterator py_objbase::end() {
	return py_holder_iterator(objbase::bs_inst_end(),&(objbase::bs_inst_end));
}

py_holder_iterator::py_holder_iterator(const py_holder_iterator &src)
	: ins(src.ins),endi(src.endi)
{}

py_holder_iterator::py_holder_iterator(const bs_objinst_holder::const_iterator &src,get_end fn)
	: ins(src),endi(fn)
{}

py_holder_iterator::~py_holder_iterator() {}

py_holder_iterator::reference py_holder_iterator::operator*() const {
	if (ins != endi())
		return py_objbase(*ins);
	return py_objbase(sp_obj(NULL));
}

py_holder_iterator::pointer py_holder_iterator::operator->() const {
	if (ins != endi())
		return py_objbase(*ins);
	return py_objbase(sp_obj(NULL));
}

py_holder_iterator &py_holder_iterator::operator++() {
	++ins;
	return *this;
}

py_holder_iterator py_holder_iterator::operator++(int) {
	py_holder_iterator tmp(*this);
	++ins;
	return tmp;
}

py_holder_iterator &py_holder_iterator::operator--() {
	--ins;
	return *this;
}

py_holder_iterator py_holder_iterator::operator--(int) {
	py_holder_iterator tmp(*this);
	--ins;
	return tmp;
}

bool py_holder_iterator::operator ==(const py_holder_iterator &ritr) const {
	return (ins == ritr.ins);
}

bool py_holder_iterator::operator !=(const py_holder_iterator &ritr) const {
	return (ins != ritr.ins);
}

const py_holder_iterator &py_holder_iterator::operator =(py_holder_iterator &ritr) {
	ins = ritr.ins;
	endi = ritr.endi;
	return *this;
}

void py_export_objbase() {
        class_ <objbase, boost::noncopyable> ("real_objbase", no_init)
          ;
	class_<py_objbase , bases<py_bs_messaging> >("objbase", no_init)
		.def("__iter__", boost::python::iterator< py_objbase >())
		.def("resolve_type",&py_objbase::bs_resolve_type)
		.def("ptr",&py_objbase::ptr)
		.def("inode",&py_objbase::inode)
		.def("fire_signal",&py_objbase::fire_signal)
		.def("get_signal_list",&py_objbase::get_signal_list)
		.def("refs", &py_objbase::refs)
		.def(self == self);

	class_<sp_obj, noncopyable>("sp_obj", no_init);

	enum_<objbase::signal_codes>("objbase_signal_codes")
		.value("on_unlock",objbase::on_unlock)
		.export_values();
}

}	//namespace blue_sky::python
}	//namespace blue_sky
