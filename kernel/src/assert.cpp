/// @file
/// @author Sergey Miryanov, Alexander Gagarin
/// @date 14.03.2017
/// @brief bs_assert impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/assert.h>
#include <bs/error.h>

#include <stdlib.h>
#include <cassert>
#include <iostream>

#ifdef _WIN32

#include <windows.h>
#include <crtdbg.h>

#if defined(_M_X64) || _WIN32 >= 0x400
#define BREAK_HERE        \
if (IsDebuggerPresent ()) \
	DebugBreak ();
#else
#define BREAK_HERE __asm { int 3 }
#endif

#else
#define BREAK_HERE __asm__ __volatile__ ("int $0x3")
#endif

namespace blue_sky { namespace bs_assert {

asserter::assert_state asserter::ask_user() const {
#ifdef _DEBUG
#if defined (_WIN32)
	int return_code = _CrtDbgReport (_CRT_ERROR, file, line, "", "Expression: %s\nValues: %s", cond_s, var_list.empty () ? "" : var_list.c_str ());

	if (return_code == -1)
		return STATE_KILL;
	else if (return_code == 0)
		return STATE_IGNORE;
	else if (return_code == 1)
		return STATE_BREAK;
#else // defined UNIX
	std::cout << "Assertion! '"
		<< cond_s
		<< "' file:'" << file
		<< "',line:" << line
		<< ". What do you want to do: [k]ill,[d]ebug,[i]gnore,ignore [a]ll? "
    << "Values: \r\n" << (var_list.empty () ? "" : var_list.c_str ())
    ;
	char return_code;
	std::cin >> return_code;

	if (return_code == 'k')
		return STATE_KILL;
	else if (return_code == 'i')
		return STATE_IGNORE;
	else if (return_code == 'd')
		return STATE_BREAK;
	else if (return_code == 'a')
		return STATE_IGNORE_ALL;
#endif	// #if defined (_WIN32)
#endif	// #ifdef _DEBUG

	return STATE_IGNORE;
}

bool asserter::handle() const {
	if (cond || ignore_all())
		return true;

	assert_state state = ask_user ();
	switch (state) {
		case STATE_KILL:
			exit (1);
			break;
		case STATE_BREAK:
			break;
		case STATE_IGNORE:
			return true;
		case STATE_IGNORE_ALL:
			ignore_all() = true;
			return true;
	}

	return false;
}

void asserter::break_here() const {
	BREAK_HERE;
}

asserter* asserter::make(bool b) {
	if (factory ())
		return factory ()->make (b, file, line, cond_s);

	static std::shared_ptr< asserter > asserter_;
	asserter_ = std::make_shared< asserter >(b, file, line, cond_s);
	return asserter_.get();
}

asserter* assert_factory::make(
	bool b, const char *file, int line, const char *cond_str
) {
	static std::shared_ptr< asserter > asserter_;
	asserter_ = std::make_shared< asserter >(b, file, line, cond_str);
	return asserter_.get();
}

assert_wrapper::assert_wrapper(asserter *pa) {
	if (pa && !pa->handle()) {
		pa->break_here ();
	}
	delete pa;
}

}} // namespace blue_sky::bs_assert

