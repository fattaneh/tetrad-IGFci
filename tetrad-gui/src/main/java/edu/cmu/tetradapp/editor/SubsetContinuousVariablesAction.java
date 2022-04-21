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

package edu.cmu.tetradapp.editor;

import edu.cmu.tetrad.data.ContinuousVariable;
import edu.cmu.tetrad.data.DataModel;
import edu.cmu.tetrad.data.DataModelList;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.util.JOptionUtils;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.util.List;

/**
 * Removes discrete columns from a data set.
 *
 * @author Joseph Ramsey
 */
final class SubsetContinuousVariablesAction extends AbstractAction {

    /**
     * The data editor.                         -
     */
    private DataEditor dataEditor;

    /**
     * Creates a new action to remove discrete columns.
     */
    public SubsetContinuousVariablesAction(DataEditor editor) {
        super("Copy Continuous Variables");

        if (editor == null) {
            throw new NullPointerException();
        }

        this.dataEditor = editor;
    }

    /**
     * Performs the action of loading a session from a file.
     */
    public void actionPerformed(ActionEvent e) {
        DataModel selectedDataModel = getDataEditor().getSelectedDataModel();

        if (selectedDataModel instanceof DataSet) {
            DataSet dataSet = (DataSet) selectedDataModel;

            List variables = dataSet.getVariables();
            int n = 0;

            for (Object variable : variables) {
                if (variable instanceof ContinuousVariable) {
                    n++;
                }
            }

            if (n == 0) {
                JOptionPane.showMessageDialog(getDataEditor(),
                        "There are no continuous variables in this data set.");
                return;
            }

            int[] indices = new int[n];
            int m = -1;

            for (int i = 0; i < variables.size(); i++) {
                if (variables.get(i) instanceof ContinuousVariable) {
                    indices[++m] = i;
                }
            }

            dataSet = dataSet.subsetColumns(indices);

            DataModelList list = new DataModelList();
            list.add(dataSet);
            getDataEditor().reset(list);
            getDataEditor().selectFirstTab();
        }
        else {
            JOptionPane.showMessageDialog(JOptionUtils.centeringComp(),
                    "Requires a tabular data set.");
        }
    }

    private DataEditor getDataEditor() {
        return dataEditor;
    }
}





