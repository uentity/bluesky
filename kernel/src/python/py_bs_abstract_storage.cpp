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

#include "py_bs_abstract_storage.h"
#include "bs_kernel.h"
#include "py_bs_exports.h"

namespace blue_sky {
namespace python {

py_bs_abstract_storage::py_bs_abstract_storage(const sp_storage &tsp) : py_objbase(sp_obj(tsp)) {
}

int py_bs_abstract_storage::open (const char *filename, int flags) {
	 return spstor.lock()->open(filename,flags);
}

int py_bs_abstract_storage::close () {
	 return spstor.lock()->close();
}

bool py_bs_abstract_storage::is_opened () const {
	 return spstor->is_opened();
}

int py_bs_abstract_storage::begin_object(const char *object_name) {
	 return spstor.lock()->begin_object(std::string(object_name));
}

int py_bs_abstract_storage::open_object(const char *object_name) {
	 return spstor.lock()->open_object(std::string(object_name));
}

int py_bs_abstract_storage::end_object() {
	 return spstor.lock()->end_object();
}

int py_bs_abstract_storage::write_int(const char *name, int value) {
	 return spstor.lock()->write(std::string(name),value);
}

int py_bs_abstract_storage::write_float(const char *name, float value) {
	 return spstor.lock()->write(std::string(name),value);
}

int py_bs_abstract_storage::write_double(const char *name, double value) {
	 return spstor.lock()->write(std::string(name),value);
}

int py_bs_abstract_storage::write_char(const char *name, char value) {
	 return spstor.lock()->write(std::string(name),value);
}

int py_bs_abstract_storage::write_void_int(const char *name, const void *value, int type) {
	 return spstor.lock()->write(std::string(name),value,type);
}

int py_bs_abstract_storage::write_int_int_void(const char *name, int type, int size, const void* data) {
	 return spstor.lock()->write(std::string(name),type,size,data);
}

int py_bs_abstract_storage::write_int_int_int_void(const char *name, int type, int rank, const int* dimensions, const void* data) {
	 return spstor.lock()->write(std::string(name),type,rank,dimensions,data);
}

int py_bs_abstract_storage::read_val_type(const char *name, void *value, int type) {
	 return spstor.lock()->read(std::string(name),value,type);
}

int py_bs_abstract_storage::read_type_data(const char *name, int type, void *data) {
	 return spstor.lock()->read(std::string(name),type,data);
}

int py_bs_abstract_storage::get_rank(const char *name) const {
	 return spstor->get_rank(std::string(name));
}

int py_bs_abstract_storage::get_dimensions(const char *name, int *dimensions) const {
	 return spstor->get_dimensions(std::string(name),dimensions);
}

void py_export_abstract_storage() {
	class_< py_bs_abstract_storage >("abstract_storage", init <const sp_storage&>())
		.def("open",&py_bs_abstract_storage::open)
		.def("close",&py_bs_abstract_storage::close)
		.def("is_opened",&py_bs_abstract_storage::is_opened)
		.def("begin_object",&py_bs_abstract_storage::begin_object)
		.def("open_object",&py_bs_abstract_storage::open_object)
		.def("end_object",&py_bs_abstract_storage::end_object)
		.def("write",&py_bs_abstract_storage::write_int)
		.def("write",&py_bs_abstract_storage::write_float)
		.def("write",&py_bs_abstract_storage::write_double)
		.def("write",&py_bs_abstract_storage::write_char)
		.def("write",&py_bs_abstract_storage::write_void_int)
		.def("write",&py_bs_abstract_storage::write_int_int_void)
		.def("write",&py_bs_abstract_storage::write_int_int_int_void)
		.def("read",&py_bs_abstract_storage::read_val_type)
		.def("read",&py_bs_abstract_storage::read_type_data)
		.def("get_rank",&py_bs_abstract_storage::get_rank)
		.def("get_dimensions",&py_bs_abstract_storage::get_dimensions);
}

}	//namespace blue_sky::python
}	//namespace blue_sky
