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

#include <iostream>
#include "bs_cube.h"
#include "bs_kernel.h"
#include "bs_report.h"
//#include "bs_plugin_common.h"

using namespace blue_sky;
using namespace std;

namespace blue_sky {

bs_cube::bs_cube(bs_type_ctor_param)
	: bs_refcounter(), objbase()
//: bs_node(sp_obj(this))
{
	bs_log &l = give_log::Instance();
	logname << "cube_test_channel" << this;
	BSOUT << "Try to create log with name " << logname.str() << bs_end;
	l.add_channel(sp_channel(new bs_channel(logname.str())));
	(*l[logname.str()].lock()) << std::string("Cube created!") << bs_end;
}

bs_cube::bs_cube(const bs_cube& src)
	: bs_refcounter(), objbase(src)
//: bs_node(sp_obj(this))
{
	*this = src;
}

bs_cube::~bs_cube()
{
	//DEBUG
	cout << "bs_cube at" << this << " dtor called, refcounter = " << refs() << endl;
	cout << " AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA " << endl;
}

void bs_cube::test()
{
//	smart_ptr<objbase> p_obj(bs_cube::bs_create_instance());
//	smart_ptr<objbase, true> p_obj1(p_obj);
//	p_obj1 = p_obj;

	std::cout << "bs_cube::test called" << std::endl;
	var_ = 0;
	bs_log &l = give_log::Instance();
	(*l[logname.str()].lock()) << std::string("Cube log destroing!") << bs_end;
	l.rem_channel(logname.str());	
}

cube_command::cube_command(bs_type_ctor_param)
	: bs_refcounter(), objbase(), combase()
{}

cube_command::cube_command(const cube_command& src)
	: bs_refcounter(), objbase(src), combase(src)
{
	*this = src;
}

sp_com cube_command::execute()
{
	cout << "Test cube_command has executed" << endl;
	return NULL;
}

void cube_command::unexecute()
{
	cout << "Test cube_command undo has executed" << endl;
}

	BLUE_SKY_TYPE_STD_CREATE(bs_cube)
	BLUE_SKY_TYPE_STD_CREATE(cube_command)

	BLUE_SKY_TYPE_STD_COPY(bs_cube)
	BLUE_SKY_TYPE_STD_COPY(cube_command)

	BLUE_SKY_TYPE_IMPL_SHORT(cube_command, objbase, "Short test bs_cube_command description")
	BLUE_SKY_TYPE_IMPL_SHORT(bs_cube, objbase, "Short test bs_cube description")
}
