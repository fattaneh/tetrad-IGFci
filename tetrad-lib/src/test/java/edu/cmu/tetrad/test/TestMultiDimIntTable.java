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

package edu.cmu.tetrad.test;

import edu.cmu.tetrad.util.MultiDimIntTable;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TestMultiDimIntTable {

    private MultiDimIntTable table;

    public void setUp() {

        int[] dims = new int[]{2, 3, 4, 5};

        table = new MultiDimIntTable(dims);
    }

    @Test
    public void testSize() {
        setUp();
        assertEquals(table.getNumCells(), 2 * 3 * 4 * 5);
    }

    @Test
    public void testIndexCalculation1() {
        setUp();
        int[] coords = new int[]{0, 0, 1, 0};
        int index = table.getCellIndex(coords);

        assertEquals(5, index);
    }

    @Test
    public void testIndexCalculation2() {
        setUp();
        int[] coords = new int[]{0, 1, 2, 0};
        int index = table.getCellIndex(coords);

        assertEquals(30, index);
    }

    @Test
    public void testCoordinateCalculation() {
        setUp();
        int[] coords = table.getCoordinates(30);

        assertEquals(1, coords[1]);
    }

    @Test
    public void testCellIncrement() {
        setUp();
        int[] coords = table.getCoordinates(30);

        table.increment(coords, 1);
        assertEquals(1, table.getValue(coords));
    }

    @Test
    public void testNumDimensions() {
        setUp();
        assertEquals(4, table.getNumDimensions());
    }
}





