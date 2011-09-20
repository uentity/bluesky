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
#include "bs_matcube.h"
#include "bs_kernel.h"

// заголовочный файл с матлабовскими функциями 
#include "engine.h"

// сам не знаю, как работают эти заклинания, но без них не удается либы подключить
#pragma comment (lib, "libeng.lib")
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libut.lib")

using namespace blue_sky;
using namespace std;

// конструктор по умолчанию
bs_matcube::bs_matcube(bs_type_ctor_param)
// хрен знает
//: bs_node(sp_obj(this))
{
	// обнуляем все, при создании сетка пустая
	sizei_ = 0;
	sizej_ = 0;
	cubeval_ = NULL;
}

// конструктор клонирующий
bs_matcube::bs_matcube(const bs_matcube& src)
// хрен знает
//: bs_node(sp_obj(this))
{
	sizei_ = 0;
	sizej_ = 0;
	cubeval_ = NULL;

	// размер текущей сетки устанавливается как размер образца
	set_size(src.sizei_, src.sizej_);
	// данные переносим
	for (int i = 0; i < sizei_; i++)
		for (int j = 0; j < sizej_; j++)
			cubeval_[i][j] = src.cubeval_[i][j];
}

// деструктор
bs_matcube::~bs_matcube()
{
	//DEBUG
	cout << "bs_matcube at" << this << " dtor called, refcounter = " << refs() << endl;
	cout << " AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA " << endl;
}

// тестовая функция
const char * bs_matcube::test()
{
	std::cout << "bs_matcube::test called" << std::endl;
	return "return bs_matcube::test";
}

// дамп содержимого в стандартный stdout
void bs_matcube::dump()
{
	std::cout << "bs_matcube::dump() called" << std::endl;

	// выводим i строчек о столбцов
	for (int i = 0; i < sizei_; i++)
	{
		for (int j = 0; j < sizej_; j++)
			std::cout << cubeval_[i][j] << " ";
		std::cout << std::endl;
	}
}

// посылка содержимого в матлаб под конкретным именем
void bs_matcube::paste_to_matlab(char* name)
{
	std::cout << "bs_matcube::paste_to_matlab('" << name << "') called" << std::endl;

	// хандл соединения с матлабом 
	Engine *ep;
	// соединяемся с матлабом (если есть запущенный матлаб с /automation , то он подхватывается, иначе запускается)
	if (!(ep = engOpen("\0"))) 
	{
		std::cout << "bs_matcube::paste_to_matlab failed: Can't start MATLAB engine" << std::endl;
		return;
	}
   
	// матлабовский массив
	mxArray* T = NULL;
	// создаем 2на2 матрицу
	T = mxCreateDoubleMatrix(sizei_, sizej_, mxREAL);
	// получаем доступ к ее данным как к double*
	double *data = (double *)mxGetPr(T);
	// записываем туда данные, учитывая как матлаб хранит данные в матрице (по столбцам)
	for (int i = 0; i < sizei_; i++)
		for (int j = 0; j < sizej_; j++)
			data[j*sizei_+i] = cubeval_[i][j];

	// кладем переменную в матлаб
	engPutVariable(ep, name, T);
	// освобождаем матрицу
	mxDestroyArray(T);
	// закрываем соединение с матлабом
	engClose(ep);
}

// получение содержимого из матлаба под конкретным именем
void bs_matcube::copy_from_matlab(char* name)
{
	std::cout << "bs_matcube::copy_from_matlab('" << name << "') called" << std::endl;

	// хандл соединения с матлабом 
	Engine *ep;
	// соединяемся с матлабом (если есть запущенный матлаб с /automation , то он подхватывается, иначе запускается)
	if (!(ep = engOpen("\0"))) 
	{
		std::cout << "bs_matcube::copy_from_matlab failed: Can't start MATLAB engine" << std::endl;
		return;
	}
   
	// матлабовский массив
	mxArray* T = NULL;
	// получим переменную из матлаба
	T = engGetVariable(ep, name);
	// проверим, что она есть
	if (T == NULL)
	{
		std::cout << "bs_matcube::copy_from_matlab failed: No variable with the name '" << name << "'" << std::endl;
		return;
	}

	// проверим, что тип дабл
	if (!mxIsDouble(T)) 
	{
		std::cout << "bs_matcube::copy_from_matlab failed: Variable '" << name << "' must have a type 'double'" << std::endl;
		return;
    }

	// проверим, что 2-мерная
	if (mxGetNumberOfDimensions(T) != 2)
	{
		std::cout << "bs_matcube::copy_from_matlab failed: Variable '" << name << "' must be a 2-dimensional array" << std::endl;
		return;
	}

	// проверим, что размеры по каждому измерению совпадают
	if ((mxGetDimensions(T))[0] != sizei_ || (mxGetDimensions(T))[1] != sizej_)	
	{
		std::cout << "bs_matcube::copy_from_matlab failed: Variable '" << name << "' has a size of (" << (mxGetDimensions(T))[0] << "," << (mxGetDimensions(T))[0] << ") instead of (" << sizei_ << "," << sizej_ << ")" << std::endl;
		return;
	}
	
	// получаем доступ к ее данным как к double*
	double *data = (double *)mxGetPr(T);
	// вытаскиваем данные
	for (int i = 0; i < sizei_; i++)
		for (int j = 0; j < sizej_; j++)
			cubeval_[i][j] = data[j*sizei_+i];

	// освобождаем матрицу
	mxDestroyArray(T);
	// закрываем соединение с матлабом
	engClose(ep);
}

// выполнение произвольной команды матлаба 
void bs_matcube::exec_in_matlab(char* str)
{
	std::cout << "bs_matcube::exec_in_matlab('" << str << "') called" << std::endl;

	// хандл соединения с матлабом 
	Engine *ep;
	// соединяемся с матлабом (если есть запущенный матлаб с /automation , то он подхватывается, иначе запускается)
	if (!(ep = engOpen("\0"))) 
	{
		std::cout << "bs_matcube::exec_in_matlab failed: Can't start MATLAB engine" << std::endl;
		return;
	}
	
	// вычисляем переданную строчку в матлабе
	engEvalString(ep, str);
	
	// закрываем соединение с матлабом
	engClose(ep);
}

// пример использования функциональности матлаба для рассчетов: увеличение всех значений на 1 
void bs_matcube::sample_func()
{
	std::cout << "bs_matcube::sample_func() called" << std::endl;
	paste_to_matlab("sample_name");
	exec_in_matlab("sample_name = sample_name + 1");
	copy_from_matlab("sample_name");
}

// изменение размера сетки
void bs_matcube::set_size(int sizei, int sizej)
{
	std::cout << "bs_matcube::setSize(" << sizei << "," << sizej << ") called" << std::endl;

	// новый массив 
	double **newarr;
	// выделяем под него память
	newarr = new double*[sizei];
	for (int i = 0; i < sizei; i++)
	{
		newarr[i] = new double[sizej];
	}
	// заполняем его нулями
	for (int i = 0; i < sizei; i++)
	{
		for (int j = 0; j < sizej; j++)
            newarr[i][j] = 0;
	}
	// если определен старый массив, то переносим из него данные
	if (cubeval_)
	{
		for (int i = 0; i < min(sizei, sizei_); i++)
			for (int j = 0; j < min(sizej, sizej_); j++)
               newarr[i][j] = cubeval_[i][j];
		// старый массив освобождаем
		for (int i = 0; i < sizei_; i++)
			delete cubeval_[i];
		delete cubeval_;
	}
	// установим новые размеры
	sizei_ = sizei;
	sizej_ = sizej;
	// подменяем ссылку на текущий массив
	cubeval_ = newarr;
}


// конструктор по умолчанию для команды
matcube_command::matcube_command(bs_type_ctor_param)
{

}

// клонирующий конструктор для команды
matcube_command::matcube_command(const matcube_command& src)
{
	*this = src;
}

// выполнение команды
sp_com matcube_command::execute()
{
	cout << "Test matcube_command has executed" << endl;
	return NULL;
}

// откат команды
void matcube_command::unexecute()
{
	cout << "Test matcube_command undo has executed" << endl;
}

// хрен знает
namespace blue_sky {
	BLUE_SKY_TYPE_STD_CREATE(bs_matcube)
	BLUE_SKY_TYPE_STD_CREATE(matcube_command)

	BLUE_SKY_TYPE_STD_COPY(bs_matcube)
	BLUE_SKY_TYPE_STD_COPY(matcube_command)

	BLUE_SKY_TYPE_IMPL_SHORT(matcube_command, objbase, "Short test bs_matmatcube_command description")
	BLUE_SKY_TYPE_IMPL_SHORT(bs_matcube, objbase, "Short test bs_matcube description")
}
