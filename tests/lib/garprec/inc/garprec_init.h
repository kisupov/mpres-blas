#ifndef __GARPREC_INIT_H__
#define __GARPREC_INIT_H__

//return MAX_PREC_WORDS
int garprecInit(const unsigned int numDigit, const int device = 0);

void garprecFinalize();

#endif 

