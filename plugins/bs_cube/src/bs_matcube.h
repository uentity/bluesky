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

#ifndef BS_MATCUBE_H_
#define BS_MATCUBE_H_

// подключаемые файлы для BS-плагина
#include "bs_object_base.h"
#include "bs_command.h"

// подключаемые файлы для BS-плагина, который будет доступен через питон (?)
#ifdef BSPY_EXPORTING_PLUGIN
#include "py_bs_tree.h"
#include "py_bs_object_base.h"
#include "py_bs_command.h"
#include "bs_plugin_common.h"
#endif

namespace blue_sky {
	namespace python {
		// хрен знает
		void register_obj();
	}

	// класс сетка - наследуется от базового класса для всех BS-объектов
	class BS_API_PLUGIN bs_matcube : public objbase
	{
		// хрен знает
		friend void python::register_obj();
		// хрен знает
		static int py_factory_index;
	public:
		// конструкторы строются автоматически директивой BLUE_SKY_TYPE_DECL(bs_matcube) ниже
		// деструктор
		~bs_matcube();

		// тестовая функция
		const char * test();
		// дамп содержимого в стандартный stdout
		void dump();

		// изменение размера сетки (выделение памяти)
		void set_size(int sizei, int sizej);

		// посылка содержимого в матлаб под конкретным именем
		void paste_to_matlab(char* name);
		// получение содержимого из матлаба под конкретным именем
		void copy_from_matlab(char* name);
		// выполнение произвольной команды матлаба 
		void exec_in_matlab(char* name);
		
		// пример использования функциональности матлаба для рассчетов: увеличение всех значений на 1 
		void sample_func();

		// хрен знает
		int get_py_factory_index();
	private:
		// размер сетки
		int sizei_,sizej_;
		// значения сетки
		double **cubeval_;
		
		// директива, регистрирующая тип (?) и создающая конструкторы
		BLUE_SKY_TYPE_DECL(bs_matcube);
	};


	// класс - команда сетки, наследуется от базового класса для всех BS-объектов и от базового класса для команд
	class BS_API_PLUGIN matcube_command : public objbase, public combase
	{
	public:
		// метод выполнения команды
		sp_com execute();
		// метод undo команды
		void unexecute();
		// тестовая функция
		void test();
		
		// хрен знает
		void dispose() const {
			delete this;
		}

		// директива, регистрирующая тип (?) и создающая конструкторы
		BLUE_SKY_TYPE_DECL(matcube_command)
	};
}

#endif
