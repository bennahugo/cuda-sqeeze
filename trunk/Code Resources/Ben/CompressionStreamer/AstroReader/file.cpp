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


#include "file.h"
namespace astroReader {
/*
 * Open a astronomy dataset
 * @params filename path to the file to be loaded (this must be a valid MeerKAT HDF5 file
 * @throws invalidFileType
 * @throws fileNotFound
 */
file::file(std::string filename):fShouldCloseFile(true)
{
  fFileHandle = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (fFileHandle < 0)
    throw astroReader::fileNotFound();
  
  //Do a validation check:
  hid_t data = H5Dopen1(fFileHandle,"/Data/correlator_data");
  if (data < 0)
    throw astroReader::invalidFileType();
  hid_t filespace = H5Dget_space(data);    /* Get filespace handle first. */
  fDimCount = H5Sget_simple_extent_ndims(filespace);
  fDimSizes = new hsize_t[fDimCount];
  H5Sget_simple_extent_dims(filespace, fDimSizes, NULL);
  if (fDimCount != 4 || fDimSizes[3] != 2) 
    throw astroReader::invalidFileType();
  H5Dclose(data);
}
/*
 * Copy constructor
 */
file::file(const file& other):fShouldCloseFile(false),fFileHandle(other.fFileHandle)
{
  fDimSizes = new hsize_t[fDimCount];
  memcpy(fDimSizes, other.fDimSizes,fDimCount*sizeof(hsize_t));
}
/*
 * Deletes heap memory and releases the file handle
 */
file::~file()
{
  if (fFileHandle >= 0){
    delete[] fDimSizes;
    fDimSizes = NULL;
    if (fShouldCloseFile) //if this is a copy of another object then that object should keep control over the file handle
      H5Fclose(fFileHandle);
  }
}
/*
 * Creates a deep copy of the other file wrapper
 */
file& file::operator=(const file& other)
{
    fFileHandle = other.fFileHandle;
    fShouldCloseFile = false;
    fDimCount = other.fDimCount;
    memcpy(fDimSizes,other.fDimSizes,fDimCount*sizeof(hsize_t));
    return *this;
}

/*
 * recursive method to print the file tree structure
 * @args currentGroup name of the current sub-tree (should be the empty string when first called)
 * @args the current level (number of tabs needed)
 */
void file::printStructure(std::string currentGroup, int level) const{
    using namespace std;
    
    if (currentGroup == "")
        currentGroup = "/";
    //Add some tabbing:
    for (int tabs = 0; tabs < level; ++tabs)
	  cout << '\t';
    cout << currentGroup << endl;
    
    //open the current group
    hid_t currentGroupHandle = H5Gopen(fFileHandle, currentGroup.c_str(), H5P_DEFAULT);
    hsize_t size;
    H5Gget_num_objs(currentGroupHandle,&size);
    
    //Depth-first print up to the leafs of the tree:
    for (hsize_t i = 0; i < size; ++i) {
	H5G_obj_t type = H5Gget_objtype_by_idx(currentGroupHandle,i);
	char * objName = new char[100];
	H5Gget_objname_by_idx(currentGroupHandle,i,objName, 100*sizeof(char));
	if (type == H5G_GROUP)
	    printStructure(currentGroup + (currentGroup != "/" ? "/" : "") + objName,level + 1);
	else { //if an object/array/native type is reached:
	  for (int tabs = 0; tabs < level+1; ++tabs)
	    cout << '\t';
	  cout << "DATA ELEMENT: " << objName << endl;
	}
	delete objName;
    }
}

int file::getDimensionCount() const
{
    return fDimCount;
}

hsize_t file::getDimensionSize(int dim) const
{
    if (dim >= fDimCount || dim < 0)
      throw arguementError();
    return fDimSizes[dim];
}

} //namespace astroReader