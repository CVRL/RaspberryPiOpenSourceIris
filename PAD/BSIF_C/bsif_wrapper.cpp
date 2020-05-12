//
//  bsif_wrapper.cpp

// Based on methods found at http://www.ee.oulu.fi/~jkannala/bsif/bsif.html

#include "BSIFFilter.hpp"

#include <Python.h> //include python api
//#include "/usr/local/Cellar/numpy/1.15.4/lib/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h"
#include "/usr/include/numpy/arrayobject.h"

static PyObject *
loadFilter(PyObject *self, PyObject *args)
{
    int size;
    int bits;

    // Load size and bits from args
    if (!PyArg_ParseTuple(args, "ii", &size, &bits)) return NULL;

    BSIFFilter newFilter;

    double *myFilter;
    // pass address of this pointer so the pointer can be dereferenced and changed
    newFilter.loadFilter(size, bits, &myFilter);

    // Convert to pyarray
    PyObject *array;
    int nd = 1;
    npy_intp dims[] = {size * size * bits};

    array = PyArray_SimpleNewFromData(nd, dims, NPY_DOUBLE, (void *)myFilter);

    return array;
};


static PyMethodDef BSIFMethods[] = {
    {"load",  loadFilter, METH_VARARGS,"Load a BSIF filter."},
     { NULL, NULL, 0, NULL }
};

static struct PyModuleDef BSIFDefinition = {
    PyModuleDef_HEAD_INIT,
    "BSIF",
    "A Python module that load a bsif filter",
    -1,
    BSIFMethods
};

PyMODINIT_FUNC
PyInit_bsif(void)
{
    import_array();
    return PyModule_Create(&BSIFDefinition);
}

