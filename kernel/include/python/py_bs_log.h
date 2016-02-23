/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _PY_BS_LOG_H
#define _PY_BS_LOG_H

#include "bs_report.h"
#include "py_bs_messaging.h"
#include <boost/python/wrapper.hpp>

namespace blue_sky {
namespace python {

/*class BS_API py_stream : public bs_stream, public boost::python::wrapper<bs_stream> {
public:
	py_stream() : spstream(this) {}
	//py_stream(py_stream *src) : spstream(src) {}
	//py_stream(const py_stream &src) : spstream(src.spstream) {}
	//py_stream(const bs_stream &src) : spstream(this) {}
	~py_stream() {BSOUT << "Refs: of " << spstream.get() << " = " << spstream.refs() << bs_end;}

	void write(const std::string &str) const;

	smart_ptr<bs_stream> spstream;
};*/

class BS_API stream_wrapper : public log::bs_stream {
public:
	stream_wrapper(const std::string &name, const boost::python::object &src) : bs_stream (name), obj(src) {}

	void write(const std::string &str) const;

private:
	boost::python::object obj;
};

class BS_API py_stream {
public:
	py_stream(const std::string &name, const boost::python::object &src) : spstream(new stream_wrapper(name, src)) {}

//private:
	sp_stream spstream;
};

class BS_API py_bs_channel {
	friend class py_bs_log;
public:
	py_bs_channel(const std::string&);
	py_bs_channel(const sp_channel&);

	void write(const char*) const;

	bool attach(const py_stream&) const;
	bool detach(const py_stream&) const;

	std::string get_name() const;

	void set_output_time() const;
	void set_wait_end() const;
	void set_auto_newline(bool);

private:
	sp_channel c;
	bool auto_newline;
};

typedef smart_ptr<bs_log> sp_log;

class BS_API py_bs_log /*: public py_bs_messaging */{
public:
	py_bs_log();

	py_bs_channel add_channel(const py_bs_channel&);
	bool rem_channel(const std::string&);

	py_bs_channel *get(const std::string &name_) const;
	const sp_channel& operator[](const std::string &name_) const;

	bool subscribe(int signal_code, const python_slot& slot) const;
	bool unsubscribe(int signal_code, const python_slot& slot) const;
	ulong num_slots(int signal_code) const;
	bool fire_signal(int signal_code, const py_objbase* param) const;
	std::vector< int > get_signal_list() const;

	std::list< std::string > get_ch_list() const;

private:
	bs_log &l;
};

class BS_API py_thread_log {
public:
	py_thread_log();

	typedef std::map<int, bs_log> mlog;
	typedef mlog::const_iterator const_iterator;
	typedef mlog::iterator iterator;

	py_bs_channel add_log_channel(const std::string&);
	bool add_log_stream(const std::string&, const py_stream&);
	bool rem_log_channel(const std::string&);
	bool rem_log_stream(const std::string&, const py_stream&);

	const py_bs_channel *get(const std::string&) const;
	const sp_channel &operator[](const std::string&) const;

private:
	thread_log &l;
};

}	//namespace blue_sky::python
}	//namespace blue_sky

#endif
