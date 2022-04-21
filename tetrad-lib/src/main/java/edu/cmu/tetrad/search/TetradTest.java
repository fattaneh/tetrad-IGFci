///////////////////////////////////////////////////////////////////////////////
// For information as to what this class does, see the Javadoc, below.       //
// Copyright (C) 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,       //
// 2007, 2008, 2009, 2010, 2014, 2015 by Peter Spirtes, Richard Scheines, Joseph   //
// Ramsey, and Clark Glymour.                                                //
//                                                                           //
// This program is free software; you can redistribute it and/or modify      //
// it under the terms of the GNU General Public License as published by      //
// the Free Software Foundation; either version 2 of the License, or         //
// (at your option) any later version.                                       //
//                                                                           //
// This program is distributed in the hope that it will be useful,           //
// but WITHOUT ANY WARRANTY; without even the implied warranty of            //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
// GNU General Public License for more details.                              //
//                                                                           //
// You should have received a copy of the GNU General Public License         //
// along with this program; if not, write to the Free Software               //
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA //
///////////////////////////////////////////////////////////////////////////////

package edu.cmu.tetrad.search;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.ICovarianceMatrix;
import edu.cmu.tetrad.graph.Node;

import java.util.List;

/**
 * Interface implemented by classes that test tetrad constraints. For the continuous case, we have a variety of tests,
 * including a distribution-free one (which may not be currently practical when the number of variables is too large).
 *
 * @author Ricardo Silva
 */

public interface TetradTest {
    public DataSet getDataSet();

    public int tetradScore(int i, int j, int k, int q);

    public boolean tetradScore3(int i, int j, int k, int q);

    public boolean tetradScore1(int i, int j, int k, int q);

    public boolean tetradHolds(int i, int j, int k, int q);

    public double tetradPValue(int i, int j, int k, int q);

    public double tetradPValue(int i1, int j1, int k1, int l1, int i2, int j2, int k2, int l2);

    public boolean oneFactorTest(int a, int b, int c, int d);

    public boolean oneFactorTest(int a, int b, int c, int d, int e);

    public boolean oneFactorTest(int a, int b, int c, int d, int e, int f);

    public boolean twoFactorTest(int a, int b, int c, int d);

    public boolean twoFactorTest(int a, int b, int c, int d, int e);

    public boolean twoFactorTest(int a, int b, int c, int d, int e, int f);

    public double getSignificance();

    public void setSignificance(double sig);

    public String[] getVarNames();

    List<Node> getVariables();

    ICovarianceMatrix getCovMatrix();


}





