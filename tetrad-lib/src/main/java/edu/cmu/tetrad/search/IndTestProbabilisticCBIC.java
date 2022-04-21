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

import edu.cmu.tetrad.data.DataModel;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.data.ICovarianceMatrix;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.IndependenceFact;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TetradMatrix;
import edu.pitt.dbmi.algo.bayesian.constraint.inference.BCInference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Uses Bayesian test using costomized Bic
 *
 * @author Fattaneh
 */
public class IndTestProbabilisticCBIC implements IndependenceTest {

    /**
     * Calculates probabilities of independence for conditional independence facts.
     */
//    private final BCInference bci;
    private BDeuScore bic;
    private int[] nodeDimensions;
    // Not
    private boolean threshold = false;

    /**
     * The data set for which conditional  independence judgments are requested.
     */
    private final DataSet data;

    /**
     * The nodes of the data set.
     */
    private List<Node> nodes;

    /**
     * Indices of the nodes.
     */
    private Map<Node, Integer> indices;

    /**
     * A map from independence facts to their probabilities of independence.
     */
    private Map<IndependenceFact, Double> H;
    private double posterior;
    private boolean verbose = false;
    
    private double cutoff = 0.5;

    //==========================CONSTRUCTORS=============================//
    /**
     * Initializes the test using a discrete data sets.
     */
    public IndTestProbabilisticCBIC(DataSet dataSet) {
        if (!dataSet.isDiscrete()) {
            throw new IllegalArgumentException("Not a discrete data set.");

        }

        this.data = dataSet;

        this.nodeDimensions = new int[dataSet.getNumColumns()];

        for (int j = 0; j < dataSet.getNumColumns(); j++) {
            DiscreteVariable variable = (DiscreteVariable) (dataSet.getVariable(j));
            int numCategories = variable.getNumCategories();
            nodeDimensions[j] = numCategories;
        }

        bic = new BDeuScore(this.data);

        nodes = dataSet.getVariables();

        indices = new HashMap<>();

        for (int i = 0; i < nodes.size(); i++) {
            indices.put(nodes.get(i), i);
        }

        this.H = new HashMap<>();
    }

    @Override
    public IndependenceTest indTestSubset(List<Node> vars) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isIndependent(Node x, Node y, List<Node> z) {
        Node[] _z = z.toArray(new Node[z.size()]);
        return isIndependent(x, y, _z);
    }

    @Override
    public boolean isIndependent(Node x, Node y, Node... z) {
        IndependenceFact key = new IndependenceFact(x, y, z);
      
//      System.out.print("key: " + key); 
        double pInd = Double.NaN;      

        if (!H.containsKey(key)) {
        	pInd = computeInd(x, y, z);
            H.put(key, pInd);
        }
        else {
        	pInd = H.get(key);
        }


        double p = pInd; 

        this.posterior = p;
//        System.out.println("ind: " + p); 
//        System.out.println("-------------"); 

        boolean ind ;
        if (this.threshold){
        	ind = (p >= cutoff);
        }
        else{
        	ind = RandomUtil.getInstance().nextDouble() < p;
        }

        if (ind) {
            return true;
        } else {
            return false;
        }
    }

	private double computeInd(Node x, Node y, Node... z) {
		double pInd = Double.NaN;
		List<Node> _z = new ArrayList<>();
        _z.add(x);
        _z.add(y);
        Collections.addAll(_z, z);
        
        Graph indBN = new EdgeListGraph(_z);
        for (Node n : z){
        	indBN.addDirectedEdge(n, x);
        	indBN.addDirectedEdge(n, y);
        }
        
        Graph depBN = new EdgeListGraph(_z);
        depBN.addDirectedEdge(x, y);
        for (Node n : z){
        	depBN.addDirectedEdge(n, x);
        	depBN.addDirectedEdge(n, y);
        }
        
        double indPrior = Math.log(0.5);
        double indScore = scoreDag(indBN, false, null, null);
        double scoreIndAll = indScore + indPrior;

        double depScore = scoreDag(depBN, true, x, y);
        double depPrior = Math.log(1 - indPrior);
        double scoreDepAll = depScore + depPrior;

        double scoreAll = lnXpluslnY(scoreIndAll, scoreDepAll);
//    	System.out.println("scoreDepAll: " + scoreDepAll);
//        System.out.println("scoreIndAll: " + scoreIndAll);
//        System.out.println("scoreAll: " + scoreAll);

        pInd = Math.exp(scoreIndAll - scoreAll);
        
        return pInd;
	}
//	private double computeInd(Node x, Node y, Node... z) {
//		double pInd = Double.NaN;
//		List<Node> allNodes = new ArrayList<>();
//        allNodes.add(x);
//        allNodes.add(y);
//        Collections.addAll(allNodes, z);
//        Graph[] indBNs = new Graph[1];
//        for (int i = 0; i < indBNs.length; i++){
//        	indBNs[i] = new EdgeListGraph(allNodes);
//        }
//        
//        for (Node n : z){
//        	indBNs[0].addDirectedEdge(n, x);
//        	indBNs[0].addDirectedEdge(n, y);
////        	indBNs[2].addDirectedEdge(n, x);
////        	indBNs[3].addDirectedEdge(n, y);
//        }
//        
//        Graph[] depBNs = new Graph[1];
//        for (int i = 0; i < depBNs.length; i++){
//        	depBNs[i] = new EdgeListGraph(allNodes);
//        }
//        
////      depBNs[0].addDirectedEdge(x, y);
////      depBNs[1].addDirectedEdge(x, y);
////    	depBNs[2].addDirectedEdge(y, x);
//    	depBNs[0].addDirectedEdge(x, y);
////    	depBNs[5].addDirectedEdge(y, x);
////    	depBNs[6].addDirectedEdge(x, y);
//
//        for (Node n : z){
////        	depBNs[1].addDirectedEdge(n, x);
////        	depBNs[2].addDirectedEdge(n, y);
////        	depBNs[3].addDirectedEdge(x, n);
////        	depBNs[3].addDirectedEdge(y, n);
//        	depBNs[0].addDirectedEdge(n, x);
//        	depBNs[0].addDirectedEdge(n, y);
////        	depBNs[5].addDirectedEdge(n, x);
////        	depBNs[6].addDirectedEdge(n, y);
//        }
//        
//        double scoreIndAll = Double.NEGATIVE_INFINITY; 
//        double indPrior = Math.log(0.5 / (indBNs.length));
//        double[] indScores = new double[indBNs.length];
//        double[] depScores = new double[depBNs.length];
//    	
//        for (int i = 0; i < indScores.length; i++){
//        	indScores[i] = scoreDag(indBNs[i]);
//        	scoreIndAll = lnXpluslnY(scoreIndAll, (indScores[i] + indPrior));
////        	System.out.println("scoreIndAll: " + scoreIndAll);
//        }
//        
//        
//        double scoreDepAll = Double.NEGATIVE_INFINITY;
//        double depPrior = Math.log(0.5 / (depBNs.length));
//        for (int i = 0; i < depScores.length; i++){
//        	depScores[i] = scoreDag(depBNs[i]);
//        	scoreDepAll = lnXpluslnY(scoreDepAll, (depScores[i] + depPrior));
////        	System.out.println("scoreDepAll: " + scoreDepAll);
//        }
//
//        double scoreAll = lnXpluslnY(scoreIndAll, scoreDepAll);
////    	System.out.println("scoreDepAll: " + scoreDepAll);
////        System.out.println("scoreIndAll: " + scoreIndAll);
////        System.out.println("scoreAll: " + scoreAll);
//
//        pInd = Math.exp(scoreIndAll - scoreAll);
//        
//        return pInd;
//	}
	
	public double scoreDag(Graph dag, boolean isDep, Node xx, Node yy) {

        double _score = 0.0;
        double penalty = 0.0;
        if (isDep){
        	penalty = 1.0 / this.nodeDimensions[this.indices.get(xx)];
//        	penalty = penalty * 1.5;
//        	System.out.println("x: " + this.nodeDimensions[this.indices.get(xx)] + ", y: " + this.nodeDimensions[this.indices.get(yy)] + ", penalty: " + penalty);
        }
        for (Node y : dag.getNodes()) {
            Set<Node> parents = new HashSet<>(dag.getParents(y));
            int parentIndices[] = new int[parents.size()];
            Iterator<Node> pi = parents.iterator();
            int count = 0;

            while (pi.hasNext()) {
                Node nextParent = pi.next();
                parentIndices[count++] = this.indices.get(nextParent);
            }

            int yIndex = this.indices.get(y);
//            if (isDep && (yIndex == this.nodeDimensions[this.indices.get(yy)])){
//                _score += this.bic.localScore(yIndex, parentIndices, penalty);
//
//            }
//            else{
            	_score += this.bic.localScore(yIndex, parentIndices);
//            }
        }

        return _score;
    }
	
    /**
     * Takes ln(x) and ln(y) as input, and returns ln(x + y)
     *
     * @param lnX is natural log of x
     * @param lnY is natural log of y
     * @return natural log of x plus y
     */
    private static final int MININUM_EXPONENT = -1022;
    protected double lnXpluslnY(double lnX, double lnY) {
        double lnYminusLnX, temp;

        if (lnY > lnX) {
            temp = lnX;
            lnX = lnY;
            lnY = temp;
        }

        lnYminusLnX = lnY - lnX;

        if (lnYminusLnX < MININUM_EXPONENT) {
            return lnX;
        } else {
            return Math.log1p(Math.exp(lnYminusLnX)) + lnX;
        }
    }
	
//    public double probConstraint(BCInference.OP op, Node x, Node y, Node[] z) {
//
//        int _x = indices.get(x) + 1;
//        int _y = indices.get(y) + 1;
//
//        int[] _z = new int[z.length + 1];
//        _z[0] = z.length;
//        for (int i = 0; i < z.length; i++) {
//            _z[i + 1] = indices.get(z[i]) + 1;
//        }
//
//        return bci.probConstraint(op, _x, _y, _z);
//    }

    @Override
    public boolean isDependent(Node x, Node y, List<Node> z) {
        Node[] _z = z.toArray(new Node[z.size()]);
        return !isIndependent(x, y, _z);
    }

    @Override
    public boolean isDependent(Node x, Node y, Node... z) {
        return !isIndependent(x, y, z);
    }

    @Override
    public double getPValue() {
        return posterior;
    }

    @Override
    public List<Node> getVariables() {
        return nodes;
    }

    @Override
    public Node getVariable(String name) {
        for (Node node : nodes) {
            if (name.equals(node.getName())) return node;
        }

        return null;
    }

    @Override
    public List<String> getVariableNames() {
        List<String> names = new ArrayList<>();

        for (Node node : nodes) {
            names.add(node.getName());
        }
        return names;
    }

    @Override
    public boolean determines(List<Node> z, Node y) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double getAlpha() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setAlpha(double alpha) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataModel getData() {
        return data;
    }

    @Override
    public ICovarianceMatrix getCov() {
        return null;
    }

    @Override
    public List<DataSet> getDataSets() {
        return null;
    }

    @Override
    public int getSampleSize() {
        return 0;
    }

    @Override
    public List<TetradMatrix> getCovMatrices() {
        return null;
    }

    @Override
    public double getScore() {
        return getPValue();
    }

    public Map<IndependenceFact, Double> getH() {
        return new HashMap<>(H);
    }

//    private double probOp(BCInference.OP type, double pInd) {
//        double probOp;
//
//        if (BCInference.OP.independent == type) {
//            probOp = pInd;
//        } else {
//            probOp = 1.0 - pInd;
//        }
//
//        return probOp;
//    }

    public double getPosterior() {
        return posterior;
    }

    @Override
    public boolean isVerbose() {
        return verbose;
    }

    @Override
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

	/**
	 * @param noRandomizedGeneratingConstraints
	 */
	public void setThreshold(boolean noRandomizedGeneratingConstraints) {
		this.threshold = noRandomizedGeneratingConstraints;
	}

	public void setCutoff(double cutoff) {
		this.cutoff = cutoff;
	}
}



