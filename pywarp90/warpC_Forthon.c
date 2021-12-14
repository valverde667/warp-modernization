/* Created by David P. Grote */
/* $Id: warpC_Forthon.c,v 1.9 2010/09/13 23:04:44 dave Exp $ */
/* This is a stub module which calls the init functions of all of            */
/* the modules that are part of WARP. This is needed since the modules       */
/* depend on each other and so must be incorporated into one shared          */
/* object file.                                                              */
/* #include "Forthon.h" */
#include <Python.h>
#define NPY_NO_DEPRECATED_API 8
#include <numpy/arrayobject.h>

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

static PyObject *ErrorObject;

/* ######################################################################### */
/* # Method list                                                             */
static struct PyMethodDef warpC_methods[] = {
  {NULL,NULL}};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  #ifdef MPIPARALLEL
  "warpCparallel", /* m_name */
  "warpCparallel", /* m_doc */
  #else
  "warpC", /* m_name */
  "warpC", /* m_doc */
  #endif
  -1,                  /* m_size */
  warpC_methods,    /* m_methods */
  NULL,                /* m_reload */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL,                /* m_free */
  };

/* ######################################################################### */
/* # The initialization function                                             */
#ifdef MPIPARALLEL
PyMODINIT_FUNC PyInit_warpCparallel(void)
#else
PyMODINIT_FUNC PyInit_warpC(void)
#endif
{
  PyObject *m, *d;
  /* PyObject *pystdout; */
  PyObject *date;
  m = PyModule_Create(&moduledef);
  d = PyModule_GetDict(m);
#ifdef MPIPARALLEL
  ErrorObject = PyErr_NewException("warpCparallel.error",NULL,NULL);
#else
  ErrorObject = PyErr_NewException("warpC.error",NULL,NULL);
#endif
  PyDict_SetItemString(d, "error", ErrorObject);

  date = PyUnicode_FromString(GITORIGINDATE);
  PyDict_SetItemString(d, "origindate", date);
  Py_XDECREF(date);

  date = PyUnicode_FromString(GITLOCALDATE);
  PyDict_SetItemString(d, "localdate", date);
  Py_XDECREF(date);

  date = PyUnicode_FromString(GITCOMMITHASH);
  PyDict_SetItemString(d, "commithash", date);
  Py_XDECREF(date);

  if (PyErr_Occurred())
    Py_FatalError("can not initialize module warpC");

  /* pystdout = PySys_GetObject("stdout"); */
  /* PyFile_WriteString("Forthon edition\n",pystdout); */

  import_array();

  return m;
}


