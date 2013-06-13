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


#ifndef STREAMCOLLECTOR_H
#define STREAMCOLLECTOR_H

#include "AstroReader/file.h"
#include "AstroReader/stride.h"
#include "AstroReader/stridefactory.h"

#include <assert.h>
//define a maximum "line rate": 
#define STREAM_SPEED 40/8*1024*1024*1024
namespace streamCollector{
  void chunkedRead(astroReader::file astroFile, int maxBlockSizeMB, int numPagesPerRead, 
				      void (*callback)(const astroReader::stride &));
}
#endif // STREAMCOLLECTOR_H
