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

import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.graph.TimeLagGraph;
import edu.cmu.tetrad.util.Parameters;
import edu.cmu.tetradapp.model.FactorAnalysisRunner;
import edu.cmu.tetradapp.model.IFgesRunner;
import edu.cmu.tetradapp.workbench.GraphWorkbench;

import javax.swing.*;
import java.awt.*;
import java.util.List;

/**
 * @author Michael Freenor
 */
public class FactorAnalysisEditor extends AbstractSearchEditor {

    //=========================CONSTRUCTORS============================//

    /**
     * Opens up an editor to let the user view the given PcRunner.
     */
    public FactorAnalysisEditor(FactorAnalysisRunner runner) {
        super(runner, "Factor Analysis");
    }

    //=============================== Public Methods ==================================//

    public Graph getGraph() {
        return getWorkbench().getGraph();
    }

    public void layoutByGraph(Graph graph) {
        getWorkbench().layoutByGraph(graph);
    }

    public Rectangle getVisibleRect() {
        return getWorkbench().getVisibleRect();
    }

    //==========================PROTECTED METHODS============================//


    /**
     * Sets up the editor, does the layout, and so on.
     */
    protected void setup(String resultLabel) {
        FactorAnalysisRunner runner = (FactorAnalysisRunner) getAlgorithmRunner();
        Graph graph = runner.getGraph();


        JTextArea display = new JTextArea(runner.getOutput());
        JScrollPane scrollPane = new JScrollPane(display);
        scrollPane.setPreferredSize(new Dimension(500, 400));
        display.setEditable(false);
        display.setFont(new Font("Monospaced", Font.PLAIN, 12));

        GraphUtils.circleLayout(graph, 225, 200, 150);
        GraphUtils.fruchtermanReingoldLayout(graph);

        GraphWorkbench workbench = new GraphWorkbench(graph);

        JScrollPane graphPane = new JScrollPane(workbench);
        graphPane.setPreferredSize(new Dimension(500, 400));

        Box box = Box.createHorizontalBox();
        box.add(scrollPane);

        box.add(Box.createHorizontalStrut(3));
        box.add(Box.createHorizontalStrut(5));
        box.add(Box.createHorizontalGlue());

        Box vBox = Box.createVerticalBox();
        vBox.add(Box.createVerticalStrut(15));
        vBox.add(box);
        vBox.add(Box.createVerticalStrut(5));
        box.add(graphPane);

        JPanel panel = new JPanel();
        panel.setLayout(new BorderLayout());
        panel.add(vBox, BorderLayout.CENTER);

        add(panel);
    }

    protected void addSpecialMenus(JMenuBar menuBar) {

    }

    public Graph getSourceGraph() {
        Graph sourceGraph = getWorkbench().getGraph();

        if (sourceGraph == null) {
            sourceGraph = getAlgorithmRunner().getSourceGraph();
        }

        return sourceGraph;
    }

    public List<String> getVarNames() {
        return (List<String>) getAlgorithmRunner().getParams().get("varNames", null);
    }

    public JPanel getToolbar() {
        return null;
    }

    //================================PRIVATE METHODS====================//

    private JComponent getIndTestParamBox() {
        Parameters params = getAlgorithmRunner().getParams();
        return getIndTestParamBox(params);
    }

    /**
     * Factory to return the correct param editor for independence test params.
     * This will go in a little box in the search editor.
     */
    private JComponent getIndTestParamBox(Parameters params) {
        if (params == null) {
            throw new NullPointerException();
        }

        if (params instanceof Parameters) {
            if (getAlgorithmRunner() instanceof IFgesRunner) {
                IFgesRunner gesRunner = ((IFgesRunner) getAlgorithmRunner());
                return new FgesIndTestParamsEditor(params, gesRunner.getType());
            }
        }

        if (getAlgorithmRunner().getSourceGraph() instanceof TimeLagGraph) {
            return new TimeSeriesIndTestParamsEditor(getAlgorithmRunner().getParams());
        }

        if (getAlgorithmRunner().getSourceGraph() instanceof Graph) {
            return new IndTestParamsEditor(params);
        }

        return new PcIndTestParamsEditor(params);
    }

    protected void doDefaultArrangement(Graph resultGraph) {
        if (getLatestWorkbenchGraph() != null) {   //(alreadyLaidOut) {
            GraphUtils.arrangeBySourceGraph(resultGraph,
                    getLatestWorkbenchGraph());
//        } else if (getKnowledge().isDefaultToKnowledgeLayout()) {
//            SearchGraphUtils.arrangeByKnowledgeTiers(resultGraph,
//                    getKnowledge());
//            alreadyLaidOut = true;
        } else {
            GraphUtils.circleLayout(resultGraph, 200, 200, 150);
//            alreadyLaidOut = true;
        }
    }

}





