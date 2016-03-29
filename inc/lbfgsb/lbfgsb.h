
#ifndef I3GULLIVER_LBFGSB_H_INCLUDED
#define I3GULLIVER_LBFGSB_H_INCLUDED

#ifdef __cplusplus 
extern "C" {
#endif

typedef int integer;
typedef float real;
typedef double doublereal;
typedef int logical;
typedef short ftnlen;

/* Subroutine */ int setulb_(integer *n, integer *m, doublereal *x, 
	doublereal *l, doublereal *u, integer *nbd, doublereal *scale,
	doublereal *f, doublereal *g, doublereal *factr, doublereal *pgtol,
	doublereal *wa, integer *iwa, char *task, integer *iprint,
	char *csave, logical *lsave, integer *isave, doublereal *dsave);

#ifdef __cplusplus 
}

#include "interface.h"

#endif

#endif
