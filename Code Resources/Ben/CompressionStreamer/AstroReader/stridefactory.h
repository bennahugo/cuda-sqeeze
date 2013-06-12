/*
    MeerKAT HDF5 Reader Data Reader
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


#ifndef STRIDEFACTORY_H
#define STRIDEFACTORY_H

#include <hdf5.h>

#include "exceptions.h"
#include "stride.h"
#include "file.h"
namespace astroReader {
class strideFactory
{
private:
  strideFactory(){}
public:
  static stride createStride(const file & astroFile, int max_timestamp_index, int min_timestamp_index,
    int max_freq_index, int min_freq_index, int max_correlation_pair_index, int min_correlation_pair_index);
};
}
#endif // STRIDEFACTORY_H
