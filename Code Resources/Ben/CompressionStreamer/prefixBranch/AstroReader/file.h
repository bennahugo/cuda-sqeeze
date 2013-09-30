/*
    MeerKAT HDF5 Reader File Handle
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


#ifndef FILE_H
#define FILE_H

#include <hdf5.h>
#include <string>
#include <iostream>
#include <cstring>
#include "exceptions.h"

namespace astroReader {
class file
{
  friend class strideFactory;
private:
    int fDimCount;
    hid_t fFileHandle;
    hsize_t* fDimSizes;
    bool fShouldCloseFile;
public:
    file(std::string filename);
    file(const file& other);
    virtual ~file();
    virtual file& operator=(const file& other);
    void printStructure(std::string currentGroup = "", int level = 0) const;
    int getDimensionCount() const;
    hsize_t getDimensionSize(int dim) const;
};
}
#endif // FILE_H
