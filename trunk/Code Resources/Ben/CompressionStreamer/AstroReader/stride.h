/*
    MeerKAT HDF5 Reader Data Holder
    Copyright (C) 2013  benjamin bennahugo@aol.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef STRIDE_H
#define STRIDE_H

#include <hdf5.h>
#include <iostream>
#include <cstring>
#include "complexPair.h"
#include "exceptions.h"
namespace astroReader{
class stride
{
  friend class strideFactory;
private:
  stride(){}
  float * fBuffer;
  hsize_t fBufferSize;
  int fmaxTimestampIndex, fminTimestampIndex, fmaxFreqIndex, fminFreqIndex, 
    fmaxCorrelationPairIndex, fminCorrelationPairIndex;
public:
    stride(const stride& other);
    virtual ~stride();
    virtual stride& operator=(const stride& other);
    void print() const;
    complexPair<float> getElement(int timeStampIndex, int frequencyIndex, int correlationPairIndex) const;
    void getTimeStampData(int timeStampIndex, float * buffer) const;
    int getTimeStampSize() const;
    int getMaxTimestampIndex() const;
    int getMinTimestampIndex() const; 
    int getMaxFreqIndex() const; 
    int getMinFreqIndex() const; 
    int getMaxCorrelationPairIndex() const;
    int getMinCorrelationPairIndex() const;
};
}
#endif // STRIDE_H
