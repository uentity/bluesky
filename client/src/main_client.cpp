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

#include "bs_kernel.h"

#include <iostream>
#include <time.h>
#include <stdio.h>

#include "bs_common.h"
#include BS_FORCE_PLUGIN_IMPORT()
#include "bs_cube_t.h"
#include BS_STOP_PLUGIN_IMPORT()

#include "bs_prop_base.h"
#include "bs_exception.h"
#include "bs_report.h"
#include "bs_tree.h"
#include "bs_shell.h"
#include "bs_array.h"

#ifdef BSPY_EXPORTING_PLUGIN
#include "py_bs_kernel.h"
#include "py_bs_object_base.h"
#include "bs_cube.h"
#include "py_bs_messaging.h"
#include "py_bs_tree.h"
#endif

//#ifdef UNIX
//#include "python2.5/Python.h"
//#else
//#include "Python.h"
//#endif

#include "boost/type_traits.hpp"
#include "boost/thread.hpp"

//#define USE_TBB_LIB

#ifdef USE_TBB_LIB
#include "tbb/task.h"
#include "tbb/task_scheduler_init.h"
using namespace tbb;
#endif


using namespace std;
using namespace blue_sky;
using namespace boost;
using namespace blue_sky::python;

kernel& k = give_kernel::Instance();

const int thr_cnt = 100;
const int how_many = 100;

class dummy : public objbase
{
	BLUE_SKY_TYPE_DECL(dummy)
	//friend objbase* create_dummy();

public:
	static std::string node_name;

	void incr() { ++cnt_; }
	void decr() { --cnt_; }
	void advance(long offset) {
		cnt_ += offset;
		//cout << "cnt_ = " << cnt_ << endl;
	}
	long current() const { return cnt_; }
	void set_cnt(long value) {
		cnt_ = value;
	}

	void test() const {};

	void dispose() const {
		//cout << "dummy object " << *name() << " is to be deleted" << endl;
		delete this;
	}

private:
	long cnt_;
};
typedef smart_ptr< dummy > sp_dummy;
typedef lsmart_ptr< sp_dummy > lsp_dummy;

class dummy_node : public bs_node {
private:
	BLUE_SKY_TYPE_DECL(dummy_node)

//	struct only_dummy : public bs_node::restrict_types {
//		const char* sort_name() const {
//			return "allow only dummy type and nodes";
//		}
//
//		bool accepts(const sp_link& l) const {
//			if(l->inode()) {
//				type_descriptor td = l->inode()->data()->bs_resolve_type();
//				if(td == dummy::bs_type() || td == bs_node::bs_type())
//					return true;
//			}
//			return false;
//		}
//
//		types_v accept_types() const {
//			types_v ret;
//			ret.push_back(dummy::bs_type());
//			return ret;
//		}
//	};

	//const char* sn_[3] = {"leaf1", "leaf2", "leaf3"};

public:
	bool validate_leafs() const {
		// check that proper subnodes are exist
		if(size() == 0) return false;
	}
};
typedef smart_ptr< dummy_node > sp_dummy_node;
typedef lsmart_ptr< sp_dummy > lsp_dummy_node;

//ctor
dummy::dummy(bs_type_ctor_param param) : cnt_(0) {}
//copy ctor
dummy::dummy(const dummy& src) : objbase(src), cnt_(src.cnt_) {}
//node_name
std::string dummy::node_name("dummy");

// implemenetation
BLUE_SKY_TYPE_STD_CREATE(dummy)
BLUE_SKY_TYPE_STD_COPY(dummy)
BLUE_SKY_TYPE_IMPL(dummy, objbase, "bs_dummy", "", "")

BLUE_SKY_TYPE_STD_CREATE(dummy_node)
BLUE_SKY_TYPE_IMPL_NOCOPY(dummy_node, objbase, "bs_dummy_node", "", "")

namespace blue_sky {
BLUE_SKY_PLUGIN_DESCRIPTOR("bs_client", "1.0.0", "", "");
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (int, str_val_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_map, 2, (bool, str_val_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (double, vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (int, vector_traits));
BS_TYPE_IMPL_T_EXT_MEM(bs_array, 2, (std::string, vector_traits));
BS_TYPE_IMPL_T_MEM(str_val_table, int);
}

struct dummy_restricter : public bs_node::restrict_types {
public:
	const char* sort_name() const { return "dummy type restricter"; }

	bool accepts(const sp_link& l) const {
		sp_dummy p_test(l->data(), bs_dynamic_cast());
		sp_node p_ifnode(l->data(), bs_dynamic_cast());
		return (p_test || p_ifnode);
	}

	types_v accept_types() const {
		types_v ret;
		ret.push_back(dummy::bs_type());
		ret.push_back(bs_node::bs_type());
		return ret;
	}
};

struct dummy_srt : public bs_node::sort_traits {
	struct dummy_key : public bs_node::sort_traits::key_type {
		dummy_key(const sp_link& l) {
			sp_dummy pd(l->data());
			if(pd) k_ = pd->current();
		}

		bool sort_order(const key_ptr& k) const {
			const dummy_key* pkey = (const dummy_key*)k.get();
			return k_ < pkey->k_;
		}

	private:
		long k_;
	};

	key_ptr key_generator(const sp_link& l) const {
		return new dummy_key(l);
	}

	const char* sort_name() const { return "dummy sort"; }

	bool accepts(const sp_link& l) const {
		return r_.accepts(l);
	}

	types_v accept_types() const {
		return r_.accept_types();
	}

private:
	dummy_restricter r_;
};

// dummy_node's ctor
dummy_node::dummy_node(bs_type_ctor_param param)
	: bs_node(param)
{
	set_sort(new dummy_srt());
	bs_node::insert(bs_node::create_node(), "leaf1", true);
	bs_node::insert(bs_node::create_node(), "leaf2", true);
	bs_node::insert(bs_node::create_node(), "leaf3", true);
}

struct dummy_incr {
	dummy_incr(const sp_dummy& d, long how_many)
		: d_(d), how_many_(how_many)
	{}

	void operator()()
	{
		//lsp_dummy ld(d_);
		//dummy* pd = const_cast< dummy* >(d_.get_ptr());
		for(long i = 0; i < how_many_; ++i) {
			d_.lock()->incr();
			//d_->name().lock()->push_back('a');
			//ld->incr();
			//lsp_dummy(d_)->incr();
		}
	}

private:
	sp_dummy d_;
	long how_many_;
};

struct dummy_decr {
	dummy_decr(const sp_dummy& d, long how_many)
		: d_(d), how_many_(how_many)
	{}

	void operator()()
	{
		//lsp_dummy ld(d_);
		//dummy* pd = const_cast< dummy* >(d_.get_ptr());
		for(long i = 0; i < how_many_; ++i) {
			d_.lock()->decr();
			//d_->name().lock()->erase(0);
			//ld->decr();
			//lsp_dummy(d_)->decr();
		}
	}

private:
	sp_dummy d_;
	long how_many_;
};

struct dummy_renamer {
	struct rename_catcher : public bs_slot {
		void execute(const sp_mobj& sender, int signal_code, const sp_obj& param) const {
			sp_link link(param);
			if(link && sp_dummy(link->data(), bs_dynamic_cast())) {
				cout << "dummy rename signal catched, new name " << link->name() << endl;
			}
		}
	};

	dummy_renamer(long how_many) : how_many_(how_many) {
		srand(clock() & 0xffff);
	}

	static ulong randUB(int upper_bound) {
		return (ulong)((double)(upper_bound - 1) * (rand() / (RAND_MAX + 1.0)));
	}

	static const char* title() {
		return "Multithreaded Rename Test";
	}

	static const char* info() {
		return "Starting many threads that renames dummy objects, resulting tree should stay consistent";
	}

	void operator()() {
		//srand(clock() & 0xffff);
		//find dummy node
		sp_node rn = k.bs_root()->node();
		bs_node::n_iterator pos = rn->find(dummy::node_name);
		if(pos == rn->end() || !bs_node::is_node(pos.data())) return;
		rn = pos->node();

		for(long i = 0; i < how_many_; ++i) {
			//do everything in single transaction
			lsmart_ptr< sp_node > guard(rn);
			//peek rundom dummy object
			pos = rn->begin();
			std::advance(pos, randUB(rn->size()));
			//check if we have a valid iterator
			if(pos == rn->end()) continue;
			//change randomly chosen letter
			string name = pos->name();
			name[randUB(name.size() + 1)] = 'a' + (char)randUB(26);
			//rename object
			rn->rename(pos, name);
		}
	}

private:
	long how_many_;
};

struct dummy_changer {
	dummy_changer(long how_many) : how_many_(how_many) {
		srand(clock() & 0xffff);
	}

	static ulong randUB(int upper_bound) {
		return (ulong)((double)(upper_bound - 1) * (rand() / (RAND_MAX + 1.0)));
	}

	static const char* title() {
		return "Multithreaded Custom Index Test";
	}

	static const char* info() {
		return "Starting many threads that changes internal state of dummy object, resulting tree should stay consistent";
	}

	void operator()() {
		for(long i = 0; i < how_many_; ++i) {
			//select random dummy object
			bs_objinst_holder::const_iterator p_inst = dummy::bs_inst_begin();
			std::advance(p_inst, randUB(dummy::bs_inst_cnt()));

			//randomly change dummy's internal value
			sp_dummy p_dum(*p_inst);
			p_dum.lock()->advance(rand() - rand());
			//cout << "changer: dummy_obj at '" << p_dum.get() << "' idx set to " << p_dum->current() << endl;
		}
	}

private:
	long how_many_;
};

struct measure_time {
	measure_time(bool go = true) {
		if(go) start();
	}

	void start() {
		sec_ = 0;
		c_ = clock();
	}

	void stop() {
		clock_t cur = clock();
		sec_ += (double)(cur - c_)/CLOCKS_PER_SEC;
	}

	double elapsed() const {
		return sec_;
	}

	ostream& operator<<(ostream& os) const {
		cout << "Time elapsed: " << sec_ << " seconds" << endl;
		return os;
	}

private:
	clock_t c_;
	double sec_;
};

ostream& operator<<(ostream& os, const measure_time& tm) {
	cout << "Time elapsed: " << tm.elapsed() << " seconds" << endl;
	return os;
}


void test_mt()
{
	cout << "------------------------------------------------------------------------" << endl;
	sp_dummy p_victim(k.create_object(dummy::bs_type()));
	//p_victim.lock()->bs_name() = "mt_victim";

	cout << "Starting many threads incrementing & decrementing counter, result should be 0" << endl;
	measure_time tm;
	thread_group g;
	for(int i = 0; i < thr_cnt; ++i) {
		//cout << i << endl;
		g.create_thread(dummy_incr(p_victim, how_many));
		g.create_thread(dummy_decr(p_victim, how_many));
	}
	//wait until all threads are finished
	g.join_all();
	tm.stop();

	cout << "OK, all threads are finished!" << endl;
	cout << "Result = " << p_victim->current() << "; Ref_cnt = " << p_victim->refs() << endl;
	cout << tm;
	//cout << "Victim name: " << p_victim->bs_name() << endl;
	cout << "test_mt finished" << endl;
	cout << "------------------------------------------------------------------------" << endl;
}

#ifdef USE_TBB_LIB
//define tasks for tbb scheduler
template< class subtask >
class task_gen : public task {
	ulong n_;
	const subtask& st_templ;

	template< class T >
	class my_task : public task {
		T w_;
	public:
		my_task(const T& t)
			: w_(t)
		{}

		task* execute() {
			cout << "worker execution started" << endl;
			w_();
			cout << "worker execution finished" << endl;
			return NULL;
		}

		~my_task() {};
	};

	typedef my_task< subtask > worker;

public:
	task_gen(const subtask& st, const ulong n) : st_templ(st), n_(n) {
		//if(n_ == 0) n_ = 1;
	}

	task* execute() {
		cout << "task_gen execution started" << endl;
		if(n_ == 1) {
			worker& w = *new(allocate_child()) worker(st_templ);
			set_ref_count(1);
			spawn(w);
			cout << "1 worker task spawned" << endl;
		}
		else if(n_ == 2) {
			worker& w1 = *new(allocate_child()) worker(st_templ);
			worker& w2 = *new(allocate_child()) worker(st_templ);
			set_ref_count(2);
			spawn(w1);
			spawn(w2);
			cout << "2 worker tasks spawned" << endl;
		}
		else {
			ulong a = n_ >> 1;
			task_gen& t1 = *new(allocate_child()) task_gen(st_templ, a);
			task_gen& t2 = *new(allocate_child()) task_gen(st_templ, n_ - a);
			set_ref_count(2);
			spawn(t1);
			spawn(t2);
		}
		cout << "task_gen execution finished" << endl;
		return NULL;
	}
};

//mt test using TBB
void tbb_test_mt() {
	cout << "------------------------------------------------------------------------" << endl;
	cout << endl << "Using TBB!" << endl;
	//init task scheduler
	tbb::task_scheduler_init tbb_ts;

	sp_dummy p_victim(k.create_object(dummy::bs_type()));
	//p_victim.lock()->bs_name() = "mt_victim";

	//start root task generator
	task_list l;
	l.push_back(*new(task::allocate_root()) task_gen< dummy_incr >(dummy_incr(p_victim, how_many), thr_cnt));
	l.push_back(*new(task::allocate_root()) task_gen< dummy_decr >(dummy_decr(p_victim, how_many), thr_cnt));

	cout << "Starting many threads incrementing & decrementing counter, result should be 0" << endl;
	measure_time tm;
	task::spawn_root_and_wait(l);
	tm.stop();

	cout << "OK, all threads are finished!" << endl;
	cout << "Result = " << p_victim->current() << "; Ref_cnt = " << p_victim->refs() << endl;
	cout << tm;
	//cout << "Victim name: " << p_victim->bs_name() << endl;
	cout << "test_mt finished" << endl;
	cout << "------------------------------------------------------------------------" << endl;
}
#endif	// USE_TBB_LIB

void test_props()
{
	cout << "------------------------------------------------------------------------" << endl;
	smart_ptr< str_val_table<int> > pb = k.create_object(str_val_table< int >::bs_type());
	str_val_table<int>::const_iterator pai = pb->begin();
	//smart_ptr< str_val_table<int> > p_b(&b);

	smart_ptr< data_table<> > ps = k.create_object(data_table<>::bs_type());
	ps.lock()->insert<int>("test", 10);
	ps.lock()->insert<bool>("bool_test", true);
	ps.lock()->insert<bool>("test", false);
	ps.lock()->insert<int>("test", 12);

	//assignment
	cout << ps->at< int >("test") << endl;
	ps.lock()->at< bool >("test") = true;
	cout << ps->at< bool >("test") << endl;

	int c = ps.lock()->ss<int>("test");
	c = ps.lock()->ss<int>("test");
	pb = ps->find_table< int >();
	//pAr = ps->find_val_table< float >();
	//b = ps.lock()->get_val_table< int >();

	smart_ptr< idx_data_table > psi = k.create_object(idx_data_table::bs_type());
	psi.lock()->insert<double>(12);
	psi.lock()->insert<int>(10, 2);

	//str_val_table< objbase > ot;
	//ot.add_item("test_object", k.create_object(dummy::type()));
	//ps.lock()->add_item< objbase >("test_object", ot.at("test_object"));

	//bs_guardian< int > c_guard(c);
	//k.global_dt().lock()->add_item<int>("test", 10);

	k.pert_idx_dt(dummy::bs_type()).lock()->insert< string >("hello");
	k.pert_idx_dt(dummy::bs_type()).lock()->insert< string >("test");
	smart_ptr< bs_array< string, vector_traits > > my_tbl = k.pert_idx_dt(dummy::bs_type()).lock()->table< string >();
	my_tbl.lock()->insert("test1");
	string s = (*my_tbl)[1];
	s = my_tbl->at(0);
	cout << "------------------------------------------------------------------------" << endl;
}

void test_objects()
{
	cout << "------------------------------------------------------------------------" << endl;
	cout << "blue_sky::conversion in action:" << endl;
	cout << conversion<objbase, objbase>::exists << endl;
	cout << conversion<combase, objbase>::exists << endl;
	cout << conversion<combase, objbase>::exists1of2way << endl;
	cout << conversion<dummy, objbase>::exists << endl;
	cout << conversion<bs_cube, objbase>::exists << endl;
	cout << conversion<bs_cube, combase>::exists << endl;

	cout << conversion<dummy, const objbase>::exists1of2way << endl;
	cout << conversion<dummy, const objbase>::exists1of2way_uc << endl;
	cout << conversion<const objbase, dummy>::exists1of2way << endl;
	cout << conversion<const objbase, dummy>::exists1of2way_uc << endl;
	cout << conversion<objbase, dummy>::exists1of2way << endl;
	cout << conversion<objbase, const dummy>::exists1of2way << endl;
	cout << conversion<objbase, const dummy>::exists1of2way_uc << endl;


	cout << "boost in action:" << endl;
	cout << is_base_of<objbase, combase>::value << endl;
	cout << is_base_and_derived<objbase, combase>::value << endl;
	cout << is_base_of<objbase, dummy>::value << endl;
	cout << is_base_of<combase, dummy>::value << endl;
	cout << is_convertible<objbase, combase>::value << endl;
	cout << is_convertible<combase, objbase>::value << endl;
	//cout << is_convertible<objbase, objbase>::value << endl;
	cout << is_convertible<objbase, const dummy>::value << endl;

	//return;

	smart_ptr<int> p_int(new int(10));
	smart_ptr<int> p_int1(p_int);
	smart_ptr<int> p_int2(p_int1);
	p_int1 = p_int;
	p_int.release();

	smart_ptr<objbase> p_o(k.create_object(dummy::bs_type()));
	//if(p_o) p_o.lock()->bs_name() = "test";

	sp_obj p_o_copy = k.create_object_copy(p_o);

	dummy* p1 = (dummy*)p_o.get();
	smart_ptr< dummy > p_do;
	p_do = p1;
	p_do = p_o;
	(*p_do).lock()->incr();

	smart_ptr< bs_cube > p_c(p1, bs_dynamic_cast());
	p_c.assign< BS_DYNAMIC_CAST >(p1);

	p_do = p_o;
	p_o = p_do;
	p_do->test();
	p_do.release();
	//((smart_ptr<dummy>&)p_o)->dummy();

	//dummy* p2 = p1;
	smart_ptr<dummy> p_do1(p1);
	p_do1 = p1;
	p_do1 = p_o;

	smart_ptr<dummy> p_dof(p_o);
	//p_dof = (smart_ptr<dummy, false>)p_do1;
	//if(p1 == p2) cout << "test";
	//if(p_do1 == p_dof)
	//	cout << "test";
	//p_do = (smart_ptr<objbase>)p_do1;

	smart_ptr< int > mt_pint(new int(20));
	smart_ptr< const int > p_cint;
	//p_int = *(smart_ptr<int>*)(&mt_pint);
	//(smart_ptr< const int >&)mt_pint = p_int;
	p_cint = mt_pint;
	//((smart_ptr< const int >&)mt_pint).release();

	smart_ptr< objbase > mt_pbs_test(p_dof);
	bs_mutex mut;
	mt_ptr< dummy > mt_ptr_test(p1, mut);
	if(mt_ptr_test == p_o) cout << "year!" << endl;
	if(mt_ptr_test == p1) cout << "year!" << endl;
	if(p1 == mt_ptr_test) cout << "year!" << endl;

	//(smart_ptr< int >&)mt_pint = p_int;
	//mt_pint.get_locked() = mt_pint.refs();

//	cout << "Dummy name assignemnt is to be performed" << endl;
//	p_o.lock()->bs_name() = "hello";
//	cout << "Test dummy object's name: " << p_o->bs_name() << endl;
	cout << "------------------------------------------------------------------------" << endl;
}

void test_plugins()
{
	k.create_object(bs_cube::bs_type());
	smart_ptr< bs_cube > p_cube(*bs_cube::bs_inst_begin());
	lsmart_ptr< smart_ptr< bs_cube > > p_cube1(p_cube);

	p_cube.release();
	//p_cube.swap(p_cube1);
	p_cube = p_cube1;

	//((smart_ptr< bs_cube >&)p_cube1).release();
	//((smart_ptr< bs_cube >&)p_cube2).release();

	//p_cube.lock()->test();
	//lsmart_ptr< smart_ptr< bs_cube > >(p_cube)->test();
	//p_cube1 = p_cube;

	p_cube1->test();

	BSOUT << "test_plugins done" << bs_end;
}

void print_loaded_types() {
	cout << "------------------------------------------------------------------------" << endl;
	cout << "List of loaded BlueSky types {" << endl;
	kernel::plugins_enum pg = k.loaded_plugins();
	kernel::types_enum tp;
	for(kernel::plugins_enum::const_iterator p = pg.begin(), p_end = pg.end(); p != p_end; ++p) {
		cout << "Plugin: " << p->name_ << ", version " << p->version_ << " {" << endl;
		tp = k.plugin_types(*p);
		for(kernel::types_enum::const_iterator t = tp.begin(), t_end = tp.end(); t != t_end; ++t) {
			cout << "	" << t->stype_ << " - " << t->short_descr_ << endl;
		}
		cout << "	}" << endl;
	}

	cout << "} end of BlueSky types list" << endl;
	cout << "------------------------------------------------------------------------" << endl;
}

void fill_dummy_node(ulong how_many = 10) {
	cout << "------------------------------------------------------------------------" << endl;
	cout << "Dummy Node Filling with Dummy Objects" << endl;
	sp_node root = k.bs_root()->node();
	//add "dummy" directory
	bs_node::n_iterator dn = root.lock()->insert(k.create_object(dummy_node::bs_type()), dummy::node_name).first;

	//position to created node
	//bs_node::n_iterator dn = root->search(string("dumb"));
	if(dn == root->end()) return;
	sp_node pdn = dn->node();
	if(!pdn) return;
	//apply sorting
	//pdn->set_sort(new dummy_srt());
	//subscribe to rename events
	sp_slot dren_slot = new dummy_renamer::rename_catcher;
	pdn->subscribe(bs_node::leaf_renamed, dren_slot);
	//apply only type restriction
	//pdn->set_sort(new dummy_restricter());
	//try to add subnode
	pdn->insert(bs_node::create_node(), std::string("test - can't be added"));

	ostringstream dname;
	measure_time tm;
	for(ulong i = 0; i < how_many; ++i) {
		dname.clear();
		dname.str("");
		dname << "dummy_" << i;
		pdn->insert(k.create_object(dummy::bs_type()), dname.str());
	}
	tm.stop();
	cout << tm;
	cout << "------------------------------------------------------------------------" << endl;
	//try to delete object
	//pdn->erase(std::string("dummy_1"));
}

void print_dummy_node(bs_node::index_type idx_t = bs_node::name_idx, bool silent = false) {
	cout << "------------------------------------------------------------------------" << endl;
	cout << "Dummy Node Contents:" << endl;
	//find dummy node
	sp_node rn = k.bs_root()->node();
	bs_node::n_iterator pos = rn->find(dummy::node_name);
	if(pos == rn->end() || !bs_node::is_node(pos.data())) {
		cout << "No Dummy Node" << endl;
		return;
	}
	rn = pos->node();
	sp_dummy p_dum;
	measure_time tm;
	for(pos = rn->begin(idx_t); pos != rn->end(idx_t); ++pos) {
		if(silent) continue;
		cout << pos->name();
		p_dum = pos->data();
		if(p_dum) {
			cout << ": " << p_dum->current();
		}
		cout << endl;
	}
	tm.stop();
	cout << "Dummy Node walking took " << tm.elapsed() << " seconds" << endl;
	cout << "------------------------------------------------------------------------" << endl;
}

void print_tree(bool silent = false) {
	cout << "------------------------------------------------------------------------" << endl;
	cout << "BlueSky tree contents {" << endl;

	sp_link leaf = k.bs_root();
	if(!silent) cout << leaf->name() << endl;
	sp_node n = leaf->node();

	deep_iterator di;
	measure_time tm;
	while(!di.is_end()) {
		if(!silent) {
	//		cout << "|" << endl;
	//		cout << "+--" << di->name();
			cout << di.full_name();
			if(di->is_node())
				cout << "(+)";
			cout << endl;
		}
		++di;
	}
	tm.stop();

//	for(bs_node::n_iterator ni = n->begin(), end = n->end(); ni != end; ++ni) {
//		cout << "|" << endl;
//		cout << "+--" << ni->name();
//		if(ni->is_node())
//			cout << "(+)";
//		cout << endl;
//	}

	cout << "} end of BlueSky tree contents" << endl;
	cout << "Tree walking took " << tm.elapsed() << " seconds" << endl;
	cout << "------------------------------------------------------------------------" << endl;
}

template< class Op >
void mt_op(long param = how_many, bool silent = false)
{
	cout << "------------------------------------------------------------------------" << endl;
	cout << Op::title() << endl;
	if(!silent) {
		cout << "Initial tree state:" << endl;
		print_tree();
	}
	cout << Op::info() << endl;
	thread_group g;
	measure_time tm;
	for(int i = 0; i < thr_cnt; ++i) {
		g.create_thread(Op(param));
	}
	//wait until all threads are finished
	g.join_all();
	tm.stop();

	cout << "OK, all threads are finished!" << endl << tm;
	if(!silent) {
		cout << "Final tree state:" << endl;
		print_tree();
	}
	cout << "------------------------------------------------------------------------" << endl;
}

//void test_python()
//{
//	Py_Initialize();
//	PyRun_InteractiveLoop(stdout,"");
//	Py_Finalize();
//}
//
//class python_slot_child : public python_slot {
//public:
//	void execute() { BSOUT << "Python slot execution!" << bs_end; }
//};

//void test_python_thread_pool() {
//    py_kernel pk;
//
//    python_slot_child ps;
//    pk.bs_root().node()->subscribe(2,ps);
//    py_bs_cube c;
//    pk.bs_root().node()->insert1(c,"asd",false);
//}

void test_ondelete() {
	struct dummy_ondelete : public bs_slot {
		 void execute(const sp_mobj& src, int signal, const sp_obj& param) const {
			 sp_dummy d(src, bs_dynamic_cast());
			 if(d)
			 	cout << "on_delete signal fired for bs_dummy object with cnt = " << d->current() << endl;
		 }
	};

	sp_dummy d = k.create_object(dummy::bs_type(), true);
	d.lock()->set_cnt(120327);
	d->subscribe(objbase::on_delete, new dummy_ondelete());
	// after exit fromthis block on_delete signal should fire
}

void test_array() {
	cout << "-----------------------------bs_array test------------------------------" << endl;
	typedef bs_array< double, vector_traits > real_array_t;
	typedef bs_array< int, vector_traits > int_array_t;
	typedef bs_array< string, vector_traits > str_array_t;
	// create arrays
	smart_ptr< real_array_t > sp_areal = k.create_object(real_array_t::bs_type());
	smart_ptr< real_array_t > sp_aint = k.create_object(int_array_t::bs_type());
	smart_ptr< real_array_t > sp_astr = k.create_object(str_array_t::bs_type());

	// fill arrays with random values
	insert_iterator< real_array_t > ii(*sp_areal.lock(), sp_areal.lock()->begin());
	for(ulong i = 0; i < 100; ++i) {
		*ii = double(rand()) / RAND_MAX;
		++ii;
	}
	// print array contents
	cout << "Real array contents:" << endl;
	for(real_array_t::const_iterator p = sp_areal->begin(), end = sp_areal->end(); p != end; ++p)
		cout << *p << ' ';
	cout << endl << "------------------------------------------------------------------------" << endl;
}

/*!
 * \brief main function
 *
 * \param argc -- number of command line arguments
 * \param argv -- command line arguments
 *
 * \return 0 if success
 */

int
main (int argc, char *argv[])
{
	//seed random number generator
	/*srand(clock() & 0xffff);
	for(int i = 0; i < 100; ++i) rand();
	srand(rand());*/

//try {
		//cout << "LoadPlugins is to be called..." << endl;
		k.LoadPlugins();

        //test_python_thread_pool();

		k.register_type(plugin_info, dummy::bs_type());

		print_loaded_types();
		//fill_dummy_node(100);
		//print_tree();

		//mt_op< dummy_changer >(50, true);
		//mt_op< dummy_renamer >(50, true);
		
		//fill_dummy_node(100);
		//print_tree();
		//mt_op< dummy_renamer >(50, true);

		//mt_op< dummy_changer >(50, true);
		//mt_op< dummy_renamer >(50, true);
		////dummy_changer(50)();
		//mt_op< dummy_changer >(50, true);
		//mt_op< dummy_renamer >(50, true);
		////dummy_renamer(50)();

		////block until all tasks are done
		//cout << "Waiting until all changes are made..." << endl;
		//k.wait_tq_empty();
		//cout << "Ok, node fully updated" << endl;
		//print_dummy_node(bs_node::custom_idx);

		k.tree_gc();

		cout << "bs_cube test" << endl;
		smart_ptr< bs_cube > c = k.create_object(bs_cube::bs_type());
		c.lock()->test();

		//test_objects();

		//test_plugins();

		//test_props();

		//test_mt();
		//print_dummy_node(bs_node::custom_idx);

		test_array();
		//test_ondelete();

#ifdef USE_TBB_LIB
		//tbb_test_mt();
#endif
		//test_python();
		//print_loaded_types();
		//print_dummy_node(bs_node::custom_idx);
//	}
//	catch(bs_exception ex) {
//		cout << "BlueSky exception: " << ex.what() << endl;
//	}
//	catch(std::exception ex) {
//		cout << "std exception: " << ex.what() << endl;
//	}
//	catch(...) {
//		cout << "unknown error!" << endl;
//	}
//
  return 0;
}
