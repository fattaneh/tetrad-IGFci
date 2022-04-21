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

package edu.cmu.tetradapp.model;

import edu.cmu.tetrad.data.DataModel;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.ImpliedOrientation;
import edu.cmu.tetrad.search.Ling;
import edu.cmu.tetrad.search.MeekRules;
import edu.cmu.tetrad.util.Parameters;
import edu.cmu.tetrad.util.TetradSerializableUtils;

import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Extends AbstractAlgorithmRunner to produce a wrapper for the GES algorithm.
 *
 * @author Ricardo Silva
 */

public class LingRunner extends AbstractAlgorithmRunner implements GraphSource,
        PropertyChangeListener {
    static final long serialVersionUID = 23L;
    private transient List<PropertyChangeListener> listeners;
    private transient Ling.StoredGraphs storedGraphs;

    //============================CONSTRUCTORS============================//

    public LingRunner(DataWrapper dataWrapper, KnowledgeBoxModel knowledgeBoxModel) {
        super(dataWrapper, new Parameters(), knowledgeBoxModel);
    }

    public LingRunner(DataWrapper dataWrapper) {
        super(dataWrapper, new Parameters(), null);
    }
    
    /**
     * Constucts a wrapper for the given EdgeListGraph.
     */
    public LingRunner(GraphSource graphWrapper, Parameters params, KnowledgeBoxModel knowledgeBoxModel) {
        super(graphWrapper.getGraph(), params, knowledgeBoxModel);
    }
    
    /**
     * Constucts a wrapper for the given EdgeListGraph.
     */
    public LingRunner(GraphSource graphWrapper, Parameters params) {
        super(graphWrapper.getGraph(), params, null);
    }
    /**
     * Generates a simple exemplar of this class to test serialization.
     *
     * @see TetradSerializableUtils
     */
    public static PcRunner serializableInstance() {
        return PcRunner.serializableInstance();
    }

    //============================PUBLIC METHODS==========================//

    /**
     * Executes the algorithm, producing (at least) a result workbench. Must be
     * implemented in the extending class.
     */

//    public void execute() {
//        DataModel source = getDataModel();
//
//        if (!(source instanceof DataSet)) {
//            throw new IllegalArgumentException("Expecting a rectangular data set.");
//        }
//
//        DataSet data = (DataSet) source;
//
//        if (!data.isContinuous()) {
//            throw new IllegalArgumentException("Expecting a continuous data set.");
//        }
//
//        Ling ling = new Ling(data);
//        Parameters searchParams = (Parameters) getParameters();
//        ling.setErrorThreshold(searchParams.getThresholdAlpha());
//        Ling.StoredGraphs graphs = ling.search();
//        Graph graph = null;
//
//        for (int i = 0; i < graphs.getNumGraphs(); i++) {
//            System.out.println(graphs.getGraph(i));
//            System.out.println(graphs.isStable(i));
//        }
//
//        for (int i = 0; i < graphs.getNumGraphs(); i++) {
//            if (graphs.isStable(i)) {
//                graph = graphs.getGraph(i);
//                break;
//            }
//        }
//
//        if (graph == null) {
//            graph = new EdgeListGraph();
//        }
//
//        setResultGraph(graph);
//        setStoredGraphs(graphs);
//
//        if (getSourceGraph() != null) {
//            DataGraphUtils.arrangeBySourceGraph(graph, getSourceGraph());
//        }
//        else {
//            DataGraphUtils.circleLayout(graph, 200, 200, 150);
//        }
//
//    }

    public void execute() {
        DataModel source = getDataModel();

        if (!(source instanceof DataSet)) {
            throw new IllegalArgumentException("Expecting a rectangular data set.");
        }

        DataSet data = (DataSet) source;

        if (!data.isContinuous()) {
            throw new IllegalArgumentException("Expecting a continuous data set.");
        }

        Ling ling = new Ling(data);
        Parameters searchParams = getParams();
        ling.setThreshold(searchParams.getDouble("threshold", 0.5));
        Ling.StoredGraphs graphs = ling.search();
        Graph graph = null;

        for (int i = 0; i < graphs.getNumGraphs(); i++) {
            System.out.println(graphs.getGraph(i));
            System.out.println(graphs.isStable(i));
        }

        for (int i = 0; i < graphs.getNumGraphs(); i++) {
            if (graphs.isStable(i)) {
                graph = graphs.getGraph(i);
                break;
            }
        }

        if (graph == null) {
            graph = new EdgeListGraph();
        }

        setResultGraph(graph);
        setStoredGraphs(graphs);

        if (getSourceGraph() != null) {
            GraphUtils.arrangeBySourceGraph(graph, getSourceGraph());
        }
        else {
            GraphUtils.circleLayout(graph, 200, 200, 150);
        }

    }

    private void setStoredGraphs(Ling.StoredGraphs graphs) {
        this.storedGraphs = graphs;
    }

    public Ling.StoredGraphs getStoredGraphs() {
        return this.storedGraphs;
    }

    public Graph getGraph() {
        return getResultGraph();
    }

    /**
     * @return the names of the triple classifications. Coordinates with
     */
    public List<String> getTriplesClassificationTypes() {
        return new LinkedList<>();
    }

    /**
     * @return the list of triples corresponding to <code>getTripleClassificationNames</code>
     * for the given node.
     * @param node The node that the classifications are for. All triple from adjacencies to this
     * node to adjacencies to this node through the given node will be considered.
     */
    public List<List<Triple>> getTriplesLists(Node node) {
        return new LinkedList<>();
    }

    public boolean supportsKnowledge() {
        return true;
    }

    public ImpliedOrientation getMeekRules() {
        MeekRules rules = new MeekRules();
        rules.setKnowledge((IKnowledge) getParams().get("knowledge", new Knowledge2()));
        return rules;
    }

    @Override
    public String getAlgorithmName() {
        return "LiNG";
    }

    private boolean isAggressivelyPreventCycles() {
        Parameters params = getParams();
        if (params instanceof Parameters) {
            return params.getBoolean("aggressivelyPreventCycles", false);
        }
        return false;
    }

    public void propertyChange(PropertyChangeEvent evt) {
        firePropertyChange(evt);
    }

    private void firePropertyChange(PropertyChangeEvent evt) {
        for (PropertyChangeListener l : getListeners()) {
            l.propertyChange(evt);
        }
    }

    private List<PropertyChangeListener> getListeners() {
        if (listeners == null) {
            listeners = new ArrayList<>();
        }
        return listeners;
    }

    public void addPropertyChangeListener(PropertyChangeListener l) {
        if (!getListeners().contains(l)) getListeners().add(l);
    }
}


