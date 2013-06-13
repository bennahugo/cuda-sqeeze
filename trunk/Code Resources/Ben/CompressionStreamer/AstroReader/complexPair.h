/*
    Complex number abstract data type
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

#ifndef COMPLEXPAIR_H
#define COMPLEXPAIR_H
template <typename T>
struct complexPair {
  float r,i;
  complexPair(T real = 0, T img = 0):r(real),i(img){}
  complexPair(const complexPair<T> & other):r(other.r),i(other.i){}
  virtual bool operator==(const complexPair<T>& rhs) const {return r == rhs.r && i == rhs.i;}
  virtual complexPair<T> & operator=(const complexPair<T> rhs) {
    r = rhs.r;
    i = rhs.i;
    return *this;
  }
};
#endif