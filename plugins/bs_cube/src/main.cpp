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

#include "bs_cube.h"
#include "bs_cube_t.h"
#include "bs_kernel.h"
//#include "bs_plugin_common.h"

#ifdef BSPY_EXPORTING_PLUGIN
#include <boost/noncopyable.hpp>
#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include "boost/python/scope.hpp"
using namespace blue_sky::python;
using namespace boost::python;
#endif

//DEBUG
//#include <iostream>

using namespace blue_sky;

typedef smart_ptr<bs_cube, true> sp_cube;

namespace blue_sky {
	BLUE_SKY_PLUGIN_DESCRIPTOR_EXT("bs_test_cube", "1.0.0", "This is a test plugin", "Long description for test plugin",
		"bs_cube")

	BLUE_SKY_REGISTER_PLUGIN_FUN
	{
		bool res = BLUE_SKY_REGISTER_TYPE(*bs_init.pd_, bs_cube_t< int >);
		res &= BLUE_SKY_REGISTER_TYPE(*bs_init.pd_, bs_cube_t< float >);
		res &= BLUE_SKY_REGISTER_TYPE(*bs_init.pd_, bs_cube_t< double >);

		res &= BLUE_SKY_REGISTER_TYPE(*bs_init.pd_, bs_cube);
		res &= BLUE_SKY_REGISTER_TYPE(*bs_init.pd_, cube_command);
			//return res;
			return true;
	}


/*
	 namespace python {
			BS_PY_CREATE_REG(bs_cube,objbase)
			BS_PY_FACTORY_REG(bs_cube)
			const int get_py_factory_index(sp_cube &sp) {return sp.lock()->get_py_factory_index();}
	 }
*/

#ifdef BSPY_EXPORTING_PLUGIN
namespace python {
	 class BS_API_PLUGIN py_bs_cube : public py_objbase	{
	 public:
			py_bs_cube();
			py_bs_cube(const py_objbase&);
			void test() { this->get_spx<bs_cube>().lock()->test();}
	 };

	/* class BS_API_PLUGIN py_bs_cube_command : public py_objbase, public py_combase {
			//class BS_API_PLUGIN py_bs_cube_command : public py_combase {
	 public:
			py_bs_cube_command();//py_objbase);
	 };		 */

	 //BSPY_OBJ_DECL_BEGIN_SHORT(bs_cube)
//	class BS_API_PLUGIN py_bs_cube : public py_bs_node {
//	public: py_bs_cube();
//			const char *test() {return "Hello, my friend!";}
//	BSPY_DECL_END

	 BSPY_COM_DECL(cube_command)

	 //BSPY_CLASSES_IMPL_SHORT(bs_cube,cube_command)
	 py_bs_cube::py_bs_cube() : py_objbase(give_kernel::Instance().create_object(bs_cube::bs_type())) {}
	 py_bs_cube::py_bs_cube(const py_objbase &obj) : py_objbase(obj) {}
	 py_cube_command::py_cube_command() : py_objbase(give_kernel::Instance().create_object(cube_command::bs_type())),
						py_combase(smart_ptr<cube_command>(sp)) {}
		/*py_bs_cube::py_bs_cube () : py_objbase(give_kernel::Instance().create_object(bs_cube::bs_type())) {}

		py_bs_cube_command::py_bs_cube_command()//py_objbase tobj)
			 //	 : py_combase(smart_ptr<bs_cube>(tobj.get_sp())) {};
		//	 : py_combase((combase*)tobj.get_sp().get()) {};
		// : py_combase((sp_com)tobj.get_sp().get()) {};
			 : py_objbase(give_kernel::Instance().create_object(cube_command::bs_type())),py_combase(smart_ptr<cube_command>(sp)) { //, py_combase(give_kernel::Instance().create_object(cube_command::bs_type())) {};
			 //spcom = smart_ptr<cube_command>(sp);
			 }*/

}

using namespace python;
/*
BOOST_PYTHON_MODULE(libbs_cube)
{
	 class_<py_bs_cube, bases <py_objbase> >("py_bs_cube",init< sp_obj& >())
			.def("test",&py_bs_cube::test);

	 class_<bs_cube, boost::noncopyable>("bs_cube",no_init);

	 class_<sp_cube, boost::noncopyable>("sp_cube",no_init)
			.add_property("index",&get_py_factory_index)
			.def("release",&sp_cube::release);

	 class_<factory_pair<sp_cube> >("bs_fp_cube", no_init)
			.def("pd",&factory_pair<sp_cube>::get_type_descriptor,return_value_policy<reference_existing_object>())
	 		.def("cf",&factory_pair<sp_cube>::create_T,return_value_policy<manage_new_object>());

	 register_obj();
	 }

*/

	 /*
BOOST_PYTHON_MODULE(libbs_cube)
{
	 class_<py_bs_cube, bases <py_objbase> >("cube")
			.def("test",&py_bs_cube::test);
	 class_<py_bs_cube_command, bases <py_objbase, py_combase> >("cube_command"); //, init <py_objbase>());
	 //class_<py_bs_cube_command, bases <py_combase> >("cube_command", init <py_objbase>());
}*/

		//BLUE_SKY_BOOST_PYTHON_MODULE_BEGIN(libbs_cube)
//BS_C_API_PLUGIN void bspy_module_init() {

//dumb struct to create new Python scope
struct py_scope_plug {};

BLUE_SKY_INIT_PY_FUN {
	//change current python scope
//	boost::python::scope outer =
//		boost::python::class_< py_scope_plug >(plugin_info.py_namespace_.c_str());

	//std::cout << "bs_cube python subsystem initialized!" << std::endl;
	BS_EXPORT_OBJBASE_CLASS_SHORT(bs_cube,"cube")
	.def(init<const py_objbase&>())
	//class_<py_bs_cube, bases< py_bs_node > >("cube")
	BS_DEF_EXPORT_SHORT2(bs_cube,test);
	BS_EXPORT_COMBASE_CLASS_SHORT(cube_command,"cube_command");
}//BLUE_SKY_BOOST_PYTHON_MODULE_END

#endif

}	//end of blue_sky namespace
