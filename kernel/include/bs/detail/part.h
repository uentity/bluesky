/// @file
/// @author uentity
/// @date 28.10.2016
/// @brief Partial function (closure) with variable arguments
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "apply.h"

/*-----------------------------------------------------------------------------
 *  Code is taken from Burst library
 *  https://github.com/izvolov/burst.git
 *  Paper reference: https://habrahabr.ru/post/313370
 *-----------------------------------------------------------------------------*/
namespace blue_sky {

//!     Результат частичного применения функции
/*!
	Функциональный объект, хранящий другой функциональный объект и первые несколько его
	аргументов.
	Вызов оператора "()" производит безусловный вызов запомненного функционального объекта
	с запомненными и новопереданными аргументами.
*/
template <typename Tuple>
struct part_fn {
	template <typename ... As>
	constexpr decltype(auto) operator ()(As && ... as) const & {
		return apply(invoke,
			std::tuple_cat(forward_tuple(t), std::forward_as_tuple(std::forward<As>(as)...)));
	}

	template <typename ... As>
	constexpr decltype(auto) operator ()(As && ... as) & {
		return apply(invoke,
			std::tuple_cat(forward_tuple(t), std::forward_as_tuple(std::forward<As>(as)...)));
	}

	template <typename ... As>
	constexpr decltype(auto) operator ()(As && ... as) && {
		return apply(invoke,
			std::tuple_cat(forward_tuple(std::move(t)), std::forward_as_tuple(std::forward<As>(as)...)));
	}

	Tuple t;
};

//!     Частичное применение функции
/*!
		Принимает некоторый функциональный объект `f` и набор первых `k` его аргументов,
	которые нужно запомнить. Сохраняет всю эту информацию внутри себя.
		Возвращает функциональный объект `f'`, зависящий от `n - k` аргументов, вызов которого
	приводит к вызову изначального функционального объекта `f` от запомненных ранее аргументов,
	а также новых аргументов, переданных объекту `f'`.
		Дано:
			f: A1 × ... × An ⟶ R
			a ∈ A1 × ... × Ak, k ≤ n
		Получаем:
			(f, a) ⟼ f'
			f': A(k + 1) × ... × An ⟶ R
*/
template <typename ... As>
constexpr auto part (As && ... as) 
	-> part_fn<decltype(std::make_tuple(std::forward<As>(as)...))>
{
	return {std::make_tuple(std::forward<As>(as)...)};
}

//template <typename Tuple, typename R>
//constexpr auto operator | (part_fn<Tuple> l, R && r) {
//	return compose(std::forward<R>(r), std::move(l));
//}

} /* namespace blue_sky */

