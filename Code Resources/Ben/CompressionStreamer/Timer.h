/*
 * Timer.h
 *
 *  Created on: 13 May 2013
 *      Author: benjamin
 */

#ifndef TIMER_H_
#define TIMER_H_
#include <time.h>
namespace Timer{
	extern timespec start, stop;
	void tic();
	double toc();
}


#endif /* TIMER_H_ */
