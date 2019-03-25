#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pytest
import py_bs_tests as tbs

class pyslot1(tbs.myslot1):
	def act(self, obj):
		print("pyslot.act: got obj", obj, obj.name(), ", use count =", obj.refs, ", pyprop = ", obj.pyprop)

class pyslot2(tbs.myslot2):
	def act(self, obj):
		print("pyslot.act: got obj", obj, obj.name(), ", use count =", obj.refs, ", pyprop = ", obj.pyprop)

def test_inheritance() :
	a = tbs.mytype()
	print("source mytype: ", a, a.name(), ", use count =", a.refs, ", pyprop = ", a.pyprop)
	print("*** test pyslot1")
	s1 = pyslot1()
	a.test_slot1(s1)
	print("*** test pyslot2")
	s2 = pyslot2()
	a.test_slot2(s2)

	assert tbs.test_objbase_pass(a)
	assert tbs.test_mytype_pass(27)

