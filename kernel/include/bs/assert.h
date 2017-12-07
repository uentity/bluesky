/// @file
/// @author Sergey Miryanov, Alexander Gagarin
/// @date 14.03.2017
/// @brief Smart assertation, based on Alexandrescu's ideas
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "setup_common_api.h"
#include <boost/lexical_cast.hpp>
#include <stdio.h>

namespace blue_sky { namespace bs_assert {

//! forward declaration
class assert_factory;

/**
 * \brief perform assert actions
 * */
class BS_API asserter {

public:
	/**
	 * \brief user reaction on assert
	 * */
	enum assert_state {
		STATE_KILL,       // ! terminate process
		STATE_BREAK,      // ! break into file where assertion failed
		STATE_IGNORE,     // ! ignore current assert
		STATE_IGNORE_ALL, // ! ignore current and all following asserts
	};

public:
	asserter(const char *file, int line, const char *cond_str)
		: cond (true)
		, file (file)
		, line (line)
		, cond_s (cond_str)
		, var_list ("")
	{
		ASSERTER_A = this;
		ASSERTER_B = this;
	}

	asserter(bool cond, const char *file, int line, const char *cond_str)
		: cond (cond)
		, file (file)
		, line (line)
		, cond_s (cond_str)
		, var_list ("")
	{
		ASSERTER_A = this;
		ASSERTER_B = this;
	}

	virtual ~asserter() {}

	/**
	 * \brief ask user what he want to do in case of false condition
	 * */
	virtual assert_state ask_user() const;
	/**
	 * \brief perform action that user select in ask_user
	 * */
	virtual bool handle() const;

	/**
	 * \brief break program execution (through int 3, for example)
	 * */
	void break_here() const;

	/**
	 * \brief pass filename, line no and string condition to asserter and
	 * return asserter object. asserter object will be used to create
	 * real asserter through call to asserter::make function.
	 * */
	static asserter workaround(const char * file_, int line_, const char *cond_str) {
		return asserter (file_, line_, cond_str);
	}

	/**
	 * \brief create real asserter through call to factory ()->make function
	 *  may return asserter that defined in python
	 * */
	asserter *make(bool cond);

	/**
	 * \brief add variable and it value to variable list
	 *  variable list will be printed in case of false condition
	 * */
	template <class T> asserter* add_var (const T &t, const std::string &name) {
		std::string str = name + " = " + boost::lexical_cast< std::string >(t) + "\n";
		var_list = var_list + str;
		return this;
	}

	/**
	 * \brief ignore all following asserts or no
	 * */
	inline bool& ignore_all () const {
		static bool ignore_all_ = false;
		return ignore_all_;
	}

	static void set_factory (assert_factory *f) {
		factory() = f;
	}

	static assert_factory*& factory () {
		static assert_factory *factory_ = 0;
		return factory_;
	}

public:
	/** 
	 * brief helpers to call add_var
	 * */
	asserter *ASSERTER_A;
	asserter *ASSERTER_B;

	bool                   cond;
	const char            *file;
	int                    line;
	const char            *cond_s;
	std::string var_list;
};

/**
 * \brief assert factory used to create an objects of assert class (or any in assert
 * hierarchy, i.e. assert_factory inherited in python and create python version 
 * of asserter)
 * */
class BS_API assert_factory {
public:
	virtual ~assert_factory () {}

	virtual asserter *make (bool b, const char *file, int line, const char *cond_str);
};

/** 
 * \brief helper class to check asserter condition, 
 * handle assert if cond is false and break if needed
 * */
struct BS_API assert_wrapper {
	assert_wrapper (asserter *pa);
};


//! define macro to check condition in debug version
#ifdef _DEBUG
#define BS_ASSERT(cond) \
	if (false) ; else     \
	bs_assert::asserter::workaround(__FILE__, __LINE__, (#cond)).make(!!(cond))->ASSERTER_A
#else
#define BS_ASSERT(cond) \
	if (true) ; else      \
	bs_assert::asserter::workaround(__FILE__, __LINE__, (#cond)).make(!!(cond))->ASSERTER_A
	//bs_assert::assert_wrapper wrapper_ = bs_assert::asserter::workaround(__FILE__, __LINE__, (#cond)).make(!!(cond))->ASSERTER_A
#endif

#ifdef _DEBUG
#define BS_ERROR(cond, caption)              \
	if (false) ; else {                      \
		BS_ASSERT ((cond));                  \
		if (!(cond))                         \
		throw bs_exception (caption, #cond); \
	}
#else
#define BS_ERROR(cond, caption)              \
	if (!(cond)) {                           \
		throw bs_exception (caption, #cond); \
	}
#endif

#define ASSERTER_A(x)         ASSERTER_OP_(x, B)
#define ASSERTER_B(x)         ASSERTER_OP_(x, A)
#define ASSERTER_OP_(x, next) ASSERTER_A->add_var ((x), #x)->ASSERTER_##next

}} // namespace blue_sky::bs_assert

