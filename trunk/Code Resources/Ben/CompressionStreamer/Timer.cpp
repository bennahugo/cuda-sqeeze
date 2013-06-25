#include "Timer.h"
timespec Timer::start;
timespec Timer::stop;
void Timer::tic(){
	clock_gettime( CLOCK_REALTIME, &start);
}
double Timer::toc(){
	clock_gettime( CLOCK_REALTIME, &stop);
	return ( stop.tv_sec - start.tv_sec )
            + (double)( stop.tv_nsec - start.tv_nsec )
              / (double)(1000000000L);
}
