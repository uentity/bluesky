#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest
import py_bs_tests as tbs

def setup_from_dtype(ar_dtype) :
	value_t = np.dtype(ar_dtype).type;
	genf = tbs.gen_vec;
	doublef = tbs.double_vec;
	if ar_dtype == np.dtype("int") :
		genf = tbs.gen_ivec;
		doublef = tbs.double_ivec;
	elif ar_dtype == np.dtype("uint") :
		genf = tbs.gen_uivec;
		doublef = tbs.double_uivec;
	return value_t, genf, doublef

def cpp2py(ar_dtype = "float") :
	value_t, genf, doublef = setup_from_dtype(ar_dtype);

	print("========")
	print("b is generated in C++")
	b = genf();
	print('b = ', b);
	assert np.all(b == np.repeat(value_t(42), len(b)))
	# resize should fail
	with pytest.raises(ValueError) :
		b.resize(12);

	b_orig = b.copy();
	doublef(b, 2);
	print('b = ', b);
	# resizing in `double_vec` should also fail (C++ -> Python[foreign data] -> C++[foreign data])
	assert len(b) == len(b_orig)
	assert np.all(b == np.repeat(value_t(84), len(b)))
	return b;

def py2cpp(ar_dtype = "float") :
	value_t, genf, doublef = setup_from_dtype(ar_dtype);

	print("========")
	print("a is created in Python")
	a = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], dtype=ar_dtype);
	print("a = ", a);

	a_orig = a.copy();
	doublef(a);
	print("doubled a = ", a);
	# resize in `double_vec` should succeed
	assert a.size == a_orig.size + 2
	assert np.all(a[:a_orig.size] == a_orig.flatten() * 2)
	assert np.all(a[-2:] == [84, 84])
	# resize in Python should also succeed
	a.resize(15);
	return a;

def test_cpp2py() :
	cpp2py("float");
	cpp2py("int");
	cpp2py("uint");

def test_py2cpp() :
	py2cpp("float");
	py2cpp("int");
	py2cpp("uint");

def test_py_nparray() :

	# resize should succeed
	print("========")
	a = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], dtype=float);
	b = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=float);
	s = a + b;
	c = tbs.test_nparray_d(a, b)
	print('c = ', c)
	assert c.size == b.size + 1;
	assert a.size == b.size + 1;
	assert c.base is a
	assert np.all(s == c[:s.size])
	assert c[-1] == np.sum(s);

	# serialization test
	#print("*** Serialization test ***")
	#tbs.test_serialization()

