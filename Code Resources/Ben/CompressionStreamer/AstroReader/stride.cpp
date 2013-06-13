/*
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2013  benjamin <email>

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


#include "stride.h"
namespace astroReader{
  
/*
 * Copy constructor
 */
stride::stride(const stride& other):fBufferSize(other.fBufferSize),
      fmaxTimestampIndex(other.fmaxTimestampIndex),fminTimestampIndex(other.fminTimestampIndex),
      fmaxFreqIndex(other.fmaxFreqIndex),fminFreqIndex(other.fminFreqIndex),
      fmaxCorrelationPairIndex(other.fmaxCorrelationPairIndex),fminCorrelationPairIndex(other.fminCorrelationPairIndex)
{
  fBuffer = new float[fBufferSize];
  memcpy(fBuffer,other.fBuffer,fBufferSize*sizeof(float));
}

/*
 * method to free the data buffer
 */
stride::~stride()
{
  delete[] fBuffer;
  fBuffer = NULL;
}

/*
 * Makes a deep copy of the other stride
 */
stride& stride::operator=(const stride& other)
{
    fBufferSize = other.fBufferSize;
    fmaxTimestampIndex = other.fmaxTimestampIndex;
    fminTimestampIndex = other.fminTimestampIndex;
    fmaxFreqIndex = other.fmaxFreqIndex;
    fminFreqIndex = other.fminFreqIndex;
    fmaxCorrelationPairIndex = other.fmaxCorrelationPairIndex;
    fminCorrelationPairIndex = other.fminCorrelationPairIndex;
    memcpy(fBuffer,other.fBuffer,fBufferSize*sizeof(float));
    return *this;
}

/*
 * method to print stride
 */
void stride::print() const{
  using namespace std;
  
    int diffTimestamp = fmaxTimestampIndex - fminTimestampIndex + 1;
    int diffFreq = fmaxFreqIndex - fminFreqIndex + 1;
    int diffCorrelation = fmaxCorrelationPairIndex - fminCorrelationPairIndex + 1;
    
    for (int ts = 0; ts < diffTimestamp; ++ts)
        for (int f = 0; f < diffFreq; ++f)
            for (int i = 0; i < diffCorrelation*2; i += 2)
                cout << "Timestamp " << ts << ", frequency index " << f << " from correlation pair " << i/2 <<
                     " (amplitude,phase): ("<<fBuffer[ts*diffFreq*diffCorrelation*2+f*diffCorrelation*2+i] << ',' <<
                     fBuffer[ts*diffFreq*diffCorrelation*2+f*diffCorrelation*2+(i+1)] << ')' << endl;
}
/*
 * Accessor for maximum correlation pair index
 */
int stride::getMaxCorrelationPairIndex() const
{
  return fmaxCorrelationPairIndex;
}
/*
 * Accessor for maximum frequency index
 */
int stride::getMaxFreqIndex() const
{
  return fmaxFreqIndex;
}
/*
 * Accessor for maximum time stamp index
 */
int stride::getMaxTimestampIndex() const
{
  return fmaxTimestampIndex;
}
/*
 * Accessor for minimum correlation pair index
 */
int stride::getMinCorrelationPairIndex() const
{
  return fminCorrelationPairIndex;
}
/*
 * Accessor for minimum frequency index
 */
int stride::getMinFreqIndex() const
{
  return fminFreqIndex;
}
/*
 * Accessor for minimum time stamp index
 */
int stride::getMminTimestampIndex() const
{
  return fminTimestampIndex;
}

}
