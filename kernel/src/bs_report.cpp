/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "bs_report.h"
#include "bs_exception.h"
#include "bs_link.h"
#include "bs_log_scribers.h"
#include "get_thread_id.h"
#include "bs_misc.h"

#include <algorithm>

using namespace std;
using namespace Loki;

namespace blue_sky {

/*-----------------------------------------------------------------
 * bs_stream implementation
 *----------------------------------------------------------------*/
namespace log {

bool bs_stream::subscribe(bs_channel &src) {
	return src.attach(sp_stream(this));
}

bool bs_stream::unsubscribe(bs_channel &src) {
	return src.detach(sp_stream(this));
}

void bs_stream::add_section (int section, int level) {
	if (sections_.find (section) != sections_.end ()) {
		bs_throw_exception ("Section already exists");
	}

	sections_.insert (std::make_pair (section, level));
}

void bs_stream::rem_section (int section) {
	section_iterator_t iter = sections_.find(section);
	if (iter != sections_.end())
		sections_.erase (iter);
}

void bs_stream::set_priority (const priority &p) {
	sections_[p.sect] = p.prior;
}

} // eof log namespace

/*-----------------------------------------------------------------
 * bs_channel implementation
 *----------------------------------------------------------------*/
bs_channel::bs_channel(const string &channel_name)
	: output_time(false)
	, wait_end(true)
	, name(channel_name)
{}

bs_channel::bs_channel(const bs_channel &src)
	: bs_refcounter(src)
{
	*this = src;
}

bs_channel::bs_channel(const bs_channel &src, const std::string &tname)
	: bs_refcounter(src)
{
	*this = src;
	name = tname;
}

const bs_channel &bs_channel::operator=(const bs_channel &src) {
	output_time     = src.output_time;
	wait_end        = src.wait_end;
	name            = src.name;
	output_streams_ = src.output_streams_;
	prefix          = src.prefix;

	return *this;
}


bool bs_channel::attach(const sp_stream &src) {
	stream_iterator_t s = find(output_streams_.begin(), output_streams_.end(), src);
	if(s == output_streams_.end()) {
		output_streams_.push_back(src);
		return true;
	}
	return false;
}

bool bs_channel::detach(const sp_stream &src) {
	stream_iterator_t s = find(output_streams_.begin(),output_streams_.end(),src);
	if (s != output_streams_.end()) {
		output_streams_.erase (s);
		return true;
	}
	return false;
}

void bs_channel::set_output_time() {
	if(output_time)
		output_time = false;
	else
		output_time = true;
}

bs_channel& bs_channel::add_section(int section, int level) {
	for (size_t i = 0, cnt = output_streams_.size (); i < cnt; ++i) {
		output_streams_[i]->add_section (section, level);
	}

	return *this;
}

bs_channel &bs_channel::rem_section(int section) {
	for (size_t i = 0, cnt = output_streams_.size (); i < cnt; ++i) {
		output_streams_[i]->rem_section (section);
	}

	return *this;
}

bs_channel& bs_channel::set_priority (const priority &tp) {
	for (size_t i = 0, cnt = output_streams_.size (); i < cnt; ++i) {
		output_streams_[i]->set_priority (tp);
	}

	return *this;
}

void bs_channel::set_prefix(const std::string &pref) {
	prefix = pref;
}

bs_channel::sp_stream_t bs_channel::get_stream (const std::string &name) const {
	for (size_t i = 0, cnt = output_streams_.size (); i < cnt; ++i) {
		if (output_streams_[i]->get_name () == name)
		{
			return output_streams_[i];
		}
	}

	bs_throw_exception (boost::format ("Unknown stream name %s") % name);
}

/*-----------------------------------------------------------------
 * locked_channel implementation
 *----------------------------------------------------------------*/
std::string locked_channel::bs_end (std::string msg) const {
	if (ch_->output_time && msg.length()) {
		std::string open_ = "[";
		std::string close_ = "]: ";

		msg = open_ + gettime () + close_ + ch_->prefix + msg + "\n";
	}
	else
		msg = msg + "\n";

	return msg;
}

/*-----------------------------------------------------------------
 * bs_log implementation
 *----------------------------------------------------------------*/
sp_channel bs_log::add_channel(const sp_channel &dest) {
	BS_ERROR (dest, "bs_log::add_channel: dest is null");

	channel_iterator_const_t itr = channel_map_.find(dest->get_name ());
	if (itr == channel_map_.end()) {
		channel_map_.insert (std::make_pair (dest->get_name (), dest));
		this->fire_signal(bs_log::log_channel_added);
		return dest;
	}
	return itr->second;
}

bool bs_log::rem_channel(const std::string &ch_name) {
	channel_iterator_t itr = channel_map_.find(ch_name);
	if (itr != channel_map_.end()) {
		channel_map_.erase(itr);
		this->fire_signal(bs_log::log_channel_removed);
		return true;
	}
	return false;
}

void bs_channel::dispose() const {
	delete this;
}

sp_channel thread_log::add_log_channel(const std::string &name) {
	unsigned long int thread = detail::get_thread_id();
	sp_log tlog = (*logs.lock())[thread];
	if (!tlog)
		tlog = new bs_log();
	return tlog.lock()->add_channel(sp_channel(new bs_channel(name)));
}

sp_channel thread_log::add_log_channel(const sp_channel &ch) {
	unsigned long int thread = detail::get_thread_id();
	sp_log tlog = (*logs.lock())[thread];
	if (!tlog)
		tlog = new bs_log();
	return tlog.lock()->add_channel(ch);
}

bool thread_log::add_log_stream(const std::string &name, const sp_stream &strm) {
	unsigned long int thread = detail::get_thread_id();
	sp_log tlog = (*logs.lock())[thread];
	if (!tlog)
		tlog = new bs_log();
	return tlog.lock()->get_locked (name, __FILE__, __LINE__).get_channel ()->attach(strm);
}

bool thread_log::rem_log_channel(const std::string &name) {
	unsigned long int thread = detail::get_thread_id();
	sp_log tlog = (*logs.lock())[thread];
	if (!tlog)
		tlog = new bs_log();
	return tlog.lock()->rem_channel(name);
}

bool thread_log::rem_log_stream(const std::string &name, const sp_stream &strm) {
	unsigned long int thread = detail::get_thread_id();
	sp_log tlog = (*logs.lock())[thread];
	if (!tlog)
		tlog = new bs_log();
	return tlog.lock()->get_locked (name, __FILE__, __LINE__).get_channel ()->detach(strm);
}

void thread_log::kill() {
	unsigned long int thread = detail::get_thread_id();
	logs.lock()->erase(thread);
}

locked_channel thread_log::operator[](const std::string &name) {
	unsigned long int thread = detail::get_thread_id();
	sp_log tlog = (*logs.lock())[thread];
	if (!tlog)
		tlog = new bs_log();
	return tlog.lock()->get_locked (name, __FILE__, __LINE__);
}

thread_log::thread_log()
	: logs(new mlog)
{}


bs_log::bs_log()
	: bs_refcounter(), bs_messaging () //(BS_SIGNAL_RANGE(bs_log))
{}


//smart pointers shouldn't delete object, because it is created as a singletone
void bs_log::dispose() const
{}

}  // eof blue_sky namespace

