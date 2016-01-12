/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "bs_shell.h"
#include "bs_kernel.h"

#include <algorithm>
#include <sstream>

using namespace std;

namespace blue_sky {

//================================= deep_iterator implementation ==========================================
class deep_iterator::di_impl {
public:
	//dummy ctor for testing
	di_impl() : leaf_(give_kernel::Instance().bs_root()->node()->begin())
	{
		path_.push_back(give_kernel::Instance().bs_root());
	}
	//copy ctor
	di_impl(const deep_iterator& di) : path_(di.pimpl_->path_), leaf_(di.pimpl_->leaf_) {}
	//construct from absolute path
	//di_impl(std::string path) {
	//}

	bool jump(const string& where) {
		path_t p;
		string leaf_name;
		//n_iterator leaf;

		istringstream fmt(where);
		if(!fmt) return false;
		while(getline(fmt, leaf_name, fmt.widen('/'))) {

		}
		return false;
	}

	bool validate_path(const path_t& context) {
		for(path_t::const_iterator p = context.begin(); p != context.end(); ++p)
			if(!(*p) || !(*p)->is_node()) return false;
		return true;
	}

	bool validated_jump(const path_t& context) {
		if(validate_path(context)) {
			path_.clear();
			copy(context.begin(), context.end(), back_inserter(path_));
			return true;
		}
		return false;
	}
	//string path parser
	//void parse(string path, )

	//access to current subroot
	sp_node parent() const {
		return path_.back()->node();
	}

	string path() const {
		string res;
		for(path_t::const_iterator p = path_.begin(), end = path_.end(); p != end; ++p) {
			res += (*p)->name();
			if((*p)->name() != "/")
				res += string("/");
		}
		return res;
	}

	string full_name() const {
		if(leaf_ != parent()->end())
			return path() + leaf_->name();
		else
			return path();
	}

	bool jump_up() {
		if(path_.size() > 1) {
			sp_link pos = path_.back();
			path_.pop_back();
			leaf_ = parent()->find(pos);
			//leaf_ = parent()->search(pos);
		}
		else if(path_.size() == 1 && leaf_ != parent()->end()) {
			//we have reached the topmost node -
			//in this and only this case leaf_ = parent()->end()
			leaf_ = parent()->end();
		}
		else
			return false;

		return true;
	}

	void operator++() {
		if(leaf_ == parent()->end()) {
			//iterator points to topmost node - go to the beginning
			leaf_ == parent()->begin();
			return;
		}
		if(leaf_->is_node() && !leaf_->node()->empty()) {
			path_.push_back(leaf_.get());
			leaf_ = parent()->begin();
			return;
		}

		//advance one position and jump up one level if we are in the end of current directory
		while(++leaf_ == parent()->end() && jump_up()) {
		}
	}

	void operator--() {
		if(leaf_ != parent()->end()) {
			if(leaf_->is_node() && !leaf_->node()->empty()) {
				path_.push_back(leaf_.get());
				leaf_ = parent()->end();
			}
			else if(leaf_ == parent()->begin()) {
				//if we are in the beginning of current directory - jump to upper level
				jump_up();
			}
		}
		else if(parent()->empty())
			//if root node is empty we cannot move anywhere
			return;
		--leaf_;
	}

	bool operator==(const di_impl& di) {
		if(path_.size() != di.path_.size()) return false;
		else return ( equal(path_.begin(), path_.end(), di.path_.begin()) &&
			leaf_ == di.leaf_ );
	}

	bool is_end() const {
		return leaf_ == parent()->end();
	}

	//members
	path_t path_;
	//sp_link leaf_;
	bs_node::n_iterator leaf_;
};

deep_iterator::deep_iterator(const sp_shell& shell)
	: pimpl_(new di_impl(shell->pos_))
{}

deep_iterator::deep_iterator()
	: pimpl_(new di_impl)
{}

deep_iterator::deep_iterator(const deep_iterator& it)
	: pimpl_(new di_impl(*it.pimpl_))
{}

deep_iterator::reference deep_iterator::operator*() const {
	return *pimpl_->leaf_;
}

deep_iterator::pointer deep_iterator::operator->() const {
	return pimpl_->leaf_.operator->();
}

deep_iterator& deep_iterator::operator++() {
	++(*pimpl_);
	return *this;
}

deep_iterator deep_iterator::operator++(int) {
	deep_iterator tmp = *this;
	return ++(*this);
	return tmp;
}

deep_iterator& deep_iterator::operator--() {
	--(*pimpl_);
	return *this;
}

deep_iterator deep_iterator::operator--(int) {
	deep_iterator tmp = *this;
	return --(*this);
	return tmp;
}

bool deep_iterator::operator==(const deep_iterator& i) const {
	return *pimpl_ == *i.pimpl_;
}

bool deep_iterator::operator!=(const deep_iterator& i) const {
	return !(*this == i);
}

string deep_iterator::full_name() const {
	return pimpl_->full_name();
}

bool deep_iterator::jump_up() {
	return pimpl_->jump_up();
}

bool deep_iterator::is_end() const {
	return pimpl_->is_end();
}

//====================================== bs_shell implementation ===================================
class bs_shell::shell_impl {
};

bs_shell::bs_shell(blue_sky::bs_type_ctor_param /*param*/)
	: pimpl_(new shell_impl)
{
	//smart_ptr< str_val_table > svt(param, bs_dynamic_cast());
	//if(!svt) {

	//}
}

BLUE_SKY_TYPE_STD_CREATE(bs_shell)
//BLUE_SKY_TYPE_STD_COPY(bs_shell)
BLUE_SKY_TYPE_IMPL_NOCOPY(bs_shell, objbase, "bs_shell", "BlueSky shell for working with objects tree", "");
//bs_shell::deep_iterator
}	//end of namespace blue_sky
