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


#include "stridefactory.h"
namespace astroReader {
stride strideFactory::createStride(const file & astroFile, int max_timestamp_index, int min_timestamp_index,
  int max_freq_index, int min_freq_index, int max_correlation_pair_index, int min_correlation_pair_index) {
    //some validation on the arguements:
    if (max_timestamp_index >= astroFile.fDimSizes[0] ||
      max_freq_index >= astroFile.fDimSizes[1] ||
      max_correlation_pair_index >= astroFile.fDimSizes[2] ||
      min_correlation_pair_index < 0 || min_freq_index < 0 || min_timestamp_index < 0)
      throw arguementError();
    
    //define stride:
    stride * ret = new stride;
    ret->fmaxCorrelationPairIndex = max_correlation_pair_index;
    ret->fmaxFreqIndex = max_freq_index;
    ret->fmaxTimestampIndex = max_timestamp_index;
    ret->fminCorrelationPairIndex = min_correlation_pair_index;
    ret->fminFreqIndex = min_freq_index;
    ret->fminTimestampIndex = min_timestamp_index;
    
    hid_t data = H5Dopen1(astroFile.fFileHandle,"/Data/correlator_data");
    hid_t filespace = H5Dget_space(data);    /* Get filespace handle first. */
   
    //Define the hyperslab to read:  
    int diffTimestamp = max_timestamp_index - min_timestamp_index + 1;
    int diffFreq = max_freq_index - min_freq_index + 1;
    int diffCorrelation = max_correlation_pair_index - min_correlation_pair_index + 1;
    hsize_t slabSize[] = {diffTimestamp,
			  diffFreq,
			  diffCorrelation,2};
    hsize_t offset[] = {min_timestamp_index,min_freq_index,min_correlation_pair_index,0};
    hsize_t count[] = {diffTimestamp,
			  diffFreq,
			  diffCorrelation,2};
    hid_t memspace =  H5Screate_simple(astroFile.fDimCount, slabSize, NULL);
    ret->fBufferSize = diffTimestamp*diffFreq*diffCorrelation*2;
    ret->fBuffer = new float[ret->fBufferSize];
    
    //Read the hyperslab:
    herr_t status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL,
				 count, NULL);
    if (status != 0)
      throw ioError();
    status = H5Dread(data, H5T_NATIVE_FLOAT, memspace, filespace,
		     H5P_DEFAULT,ret->fBuffer);
    if (status != 0)
      throw ioError();
    H5Dclose(data);
    return *ret;
}
} //namespace astroReader