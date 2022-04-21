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

import edu.cmu.tetrad.data.BoxDataSet;
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
 * Uses BCInference by Cooper and Bui to calculate probabilistic conditional independence judgments.
 *
 * @author Fattaneh Jabbari 9/2019
 */
public class IndTestProbabilisticISBDeu implements IndependenceTest {

	private boolean threshold = false;

	/**
	 * The data set for which conditional  independence judgments are requested.
	 */
	private final DataSet data;
	private final DataSet test;
	private final int[][] data_array;
	private final int[][] test_array;
	private double prior = 0.5;
	/**
	 * The nodes of the data set.
	 */
	private List<Node> nodes;

	private final int[] nodeDimensions ;

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
	public IndTestProbabilisticISBDeu(DataSet dataSet, DataSet test, double prior) {
		if (!dataSet.isDiscrete()) {
			throw new IllegalArgumentException("Not a discrete data set.");

		}

		this.prior = prior;
		this.data = dataSet;
		this.test = test;
		
		//  convert the data and the test case to an array
		this.test_array = new int[this.test.getNumRows()][this.test.getNumColumns()];
		for (int i = 0; i < test.getNumRows(); i++) {
			for (int j = 0; j < test.getNumColumns(); j++) {
				this.test_array[i][j] = test.getInt(i, j);
			}
		}

		this.data_array = new int[dataSet.getNumRows()][dataSet.getNumColumns()];

		for (int i = 0; i < dataSet.getNumRows(); i++) {
			for (int j = 0; j < dataSet.getNumColumns(); j++) {
				this.data_array[i][j] = dataSet.getInt(i, j);
			}
		}


		this.nodeDimensions = new int[dataSet.getNumColumns() + 2];

		for (int j = 0; j < dataSet.getNumColumns(); j++) {
			DiscreteVariable variable = (DiscreteVariable) (dataSet.getVariable(j));
			int numCategories = variable.getNumCategories();
			this.nodeDimensions[j + 1] = numCategories;
		}

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


		double pInd = Double.NaN;      

		if (!H.containsKey(key)) {

			// convert set z to an array of indicies
			int[] _z = new int[z.length];
			for (int i = 0; i < z.length; i++) {
				_z[i] = indices.get(z[i]);
			}

			if (_z.length == 0){
				BDeuScoreWOprior bic = new BDeuScoreWOprior(this.data);
				pInd = computeInd(bic, this.prior, x, y, z);
//				pInd = computeIndWithMultipleStructures(bic, x, y, z);
			}

			else{				
				// split the data based on array _z
				DataSet data_is = new BoxDataSet((BoxDataSet) this.data);
				DataSet data_rest = new BoxDataSet((BoxDataSet) this.data);
				splitData(data_is, data_rest, _z);
				
				BDeuScoreWOprior bic_res = new BDeuScoreWOprior(data_rest);
				double priorP = computeInd(bic_res, this.prior, x, y, z);
				
				BDeuScoreWOprior bic_is = new BDeuScoreWOprior(data_is);
				pInd = computeInd(bic_is, priorP, x, y, z);

//				// compute BSC based on D that matches values of _z in the test case
//				if(data_is.getNumRows() > 0){ 
//					BDeuScoreWOprior bic_is = new BDeuScoreWOprior(data_is);
//					pInd = computeInd(bic_is, priorP, x, y, z);
//				}
//				else{
//					pInd = priorP;
//				}

			}

			H.put(key, pInd);

		}else {
			pInd = H.get(key);
		}

		//        System.out.println("pInd_old: " + pInd_old);
		//        System.out.println("pInd: " + pInd);
		//        System.out.println("--------------------");
		double p = pInd; 

		this.posterior = p;

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



	private double computeInd(BDeuScoreWOprior bic, double prior, Node x, Node y, Node... z) {
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
		double indPrior = Math.log(prior);
		double indScore = scoreDag(indBN,bic);
		//      double indScore = scoreDag(indBN, bic, false, null, null);
		double scoreIndAll = indScore + indPrior;


		double depScore = scoreDag(depBN, bic);
		//      double depScore = scoreDag(depBN, bic, true, x, y);
		double depPrior = Math.log(1 - indPrior);
		double scoreDepAll = depScore + depPrior;

		double scoreAll = lnXpluslnY(scoreIndAll, scoreDepAll);
		//  	System.out.println("scoreDepAll: " + scoreDepAll);
		//      System.out.println("scoreIndAll: " + scoreIndAll);
		//      System.out.println("scoreAll: " + scoreAll);

		pInd = Math.exp(scoreIndAll - scoreAll);

		return pInd;
	}
	

	//        double indPrior = Math.log(0.5);
	//        double indScore = scoreDag(indBN, bic_is);
	//        double scoreIndAll = indScore + indPrior;
	//
	//        
	//        double depScore = scoreDag(depBN, bic_is);
	//        double depPrior = Math.log(1 - indPrior);
	//        double scoreDepAll = depScore + depPrior;
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
	private void splitData(DataSet data_xy, DataSet data_rest, int[] parents){
		int sampleSize = data.getNumRows();

		Set<Integer> rows_is = new HashSet<>();
		Set<Integer> rows_res = new HashSet<>();
		
		for (int i = 0; i < data.getNumRows(); i++){
			rows_res.add(i);
		}

//		for(IndependenceFact f : H_xy.keySet()){

			for (int i = 0; i < sampleSize; i++){
				int[] parentValuesTest = new int[parents.length];
				int[] parentValuesCase = new int[parents.length];

				for (int p = 0; p < parents.length ; p++){
					parentValuesTest[p] =  test_array[0][parents[p]];
					parentValuesCase[p] = data_array[i][parents[p]];
				}

				if (Arrays.equals(parentValuesCase, parentValuesTest)){
					rows_is.add(i);
					rows_res.remove(i);
				}		
			}
//		}

		int[] is_array = new int[rows_is.size()];
		int c = 0;
		for(int row : rows_is) is_array[c++] = row;
		
		int[] res_array = new int[rows_res.size()];
		c = 0;
		for(int row : rows_res) res_array[c++] = row;
		
		Arrays.sort(is_array);
		Arrays.sort(res_array);
		
		data_xy.removeRows(res_array);
		data_rest.removeRows(is_array);
		//		System.out.println("data_xy: " + data_xy.getNumRows());
		//		System.out.println("data_rest: " + data_rest.getNumRows());

	}

//	private void splitData(int[] parents){
//
//		int sampleSize = this.data.getNumRows();
//		int numVariables = this.data.getNumColumns();
////		ArrayList<Integer> rows_is = new ArrayList<>();
////		ArrayList<Integer> rows_res = new ArrayList<>();
//		Set<Integer> rows_is = new HashSet<>();
//		Set<Integer> rows_res = new HashSet<>();
//		for (int i = 0; i < data.getNumRows(); i++){
//			rows_res.add(i);
//		}
//
//		for (int i = 0; i < sampleSize; i++){
//			int[] parentValuesTest = new int[parents.length];
//			int[] parentValuesCase = new int[parents.length];
//
//			for (int p = 0; p < parents.length ; p++){
//				parentValuesTest[p] =  test_array[0][parents[p]];
//				parentValuesCase[p] = data_array[i][parents[p]];
//			}
//			int [] row = new int[numVariables];
//			for (int j = 0; j < numVariables; j++){				
//				row[j] = data_array[i][j];
//			}
//			if (Arrays.equals(parentValuesCase, parentValuesTest)){
//				rows_is.add(i);
//				rows_res.remove(i);
//			}		
//		}
//
//		//		this.data_is = new BoxDataSet(((BoxDataSet) this.data).getDataBox(), this.data.getVariables()); 
//		this.data_is = new BoxDataSet((BoxDataSet)this.data);
//
//		//		this.data_res = new BoxDataSet(((BoxDataSet) this.data).getDataBox(), this.data.getVariables());
//		this.data_res = new BoxDataSet((BoxDataSet)this.data);
//		
////		System.out.println("is :" + rows_is);
////		System.out.println("res :" + rows_res);
//		int[] is_array = new int[rows_is.size()];
//		int c = 0;
//		for(int row : rows_is) is_array[c++] = row;
//		int[] res_array = new int[rows_res.size()];
//		c = 0;
//		for(int row : rows_res) res_array[c++] = row;
//		Arrays.sort(is_array);
//		Arrays.sort(res_array);
//		this.data_is.removeRows(res_array);
//		this.data_res.removeRows(is_array);
//		//		System.out.println("data     :" + this.data.getNumRows());
//		//		System.out.println("data is  :" + this.data_is.getNumRows());
//		//		System.out.println("data res :" + this.data_res.getNumRows());
//	}

	public Map<IndependenceFact, Double> groupbyXYI(Map<IndependenceFact, Double> H, Node x, Node y){
		Map<IndependenceFact, Double> H_xy = new HashMap<IndependenceFact, Double>();
		for (IndependenceFact k : H.keySet()){					
			if ((k.getX().equals(x) && k.getY().equals(y) && k.getZ().size() > 0) ||(k.getX().equals(y) && k.getY().equals(x) && k.getZ().size() > 0)){
				H_xy.put(k, H.get(k));
			}
		}
		return H_xy;
	}
	public Map<IndependenceFact, Double> groupbyXYP(Map<IndependenceFact, Double> H, Node x, Node y){
		Map<IndependenceFact, Double> H_xy = new HashMap<IndependenceFact, Double>();
		for (IndependenceFact k : H.keySet()){					
			if ((k.getX().equals(x) && k.getY().equals(y)) ||(k.getX().equals(y) && k.getY().equals(x))){
				H_xy.put(k, H.get(k));
			}
		}
		return H_xy;
	}

	public void splitDatabyXY(DataSet data, DataSet data_xy, DataSet data_rest, Map<IndependenceFact, Double> H_xy){

		int sampleSize = data.getNumRows();

		Set<Integer> rows_is = new HashSet<>();
		Set<Integer> rows_res = new HashSet<>();
		for (int i = 0; i < data.getNumRows(); i++){
			rows_res.add(i);
		}

		for(IndependenceFact f : H_xy.keySet()){
			Node[] z = f.getZ().toArray(new Node[f.getZ().size()]);
			int[] parents = new int[z.length];
			for (int i = 0; i < z.length; i++) {
				parents[i] = indices.get(z[i]);
			}

			for (int i = 0; i < sampleSize; i++){
				int[] parentValuesTest = new int[parents.length];
				int[] parentValuesCase = new int[parents.length];

				for (int p = 0; p < parents.length ; p++){
					parentValuesTest[p] =  test_array[0][parents[p]];
					parentValuesCase[p] = data_array[i][parents[p]];
				}

				if (Arrays.equals(parentValuesCase, parentValuesTest)){
					rows_is.add(i);
					rows_res.remove(i);
				}		
			}
		}

		int[] is_array = new int[rows_is.size()];
		int c = 0;
		for(int row : rows_is) is_array[c++] = row;
		int[] res_array = new int[rows_res.size()];
		c = 0;
		for(int row : rows_res) res_array[c++] = row;
		Arrays.sort(is_array);
		Arrays.sort(res_array);
		data_xy.removeRows(res_array);
		data_rest.removeRows(is_array);
		//		System.out.println("data_xy: " + data_xy.getNumRows());
		//		System.out.println("data_rest: " + data_rest.getNumRows());

	}

	public double scoreDag(Graph dag, BDeuScoreWOprior bic_is) {

		double _score = 0.0;

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
			_score += bic_is.localScore(yIndex, parentIndices);
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




/////////////////////////////////////////////////////////////////////////////////
//// For information as to what this class does, see the Javadoc, below.       //
//// Copyright (C) 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,       //
//// 2007, 2008, 2009, 2010, 2014, 2015 by Peter Spirtes, Richard Scheines, Joseph   //
//// Ramsey, and Clark Glymour.                                                //
////                                                                           //
//// This program is free software; you can redistribute it and/or modify      //
//// it under the terms of the GNU General Public License as published by      //
//// the Free Software Foundation; either version 2 of the License, or         //
//// (at your option) any later version.                                       //
////                                                                           //
//// This program is distributed in the hope that it will be useful,           //
//// but WITHOUT ANY WARRANTY; without even the implied warranty of            //
//// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
//// GNU General Public License for more details.                              //
////                                                                           //
//// You should have received a copy of the GNU General Public License         //
//// along with this program; if not, write to the Free Software               //
//// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA //
/////////////////////////////////////////////////////////////////////////////////
//
//package edu.cmu.tetrad.search;
//
//import edu.cmu.tetrad.data.BoxDataSet;
//import edu.cmu.tetrad.data.DataModel;
//import edu.cmu.tetrad.data.DataSet;
//import edu.cmu.tetrad.data.DiscreteVariable;
//import edu.cmu.tetrad.data.ICovarianceMatrix;
//import edu.cmu.tetrad.graph.EdgeListGraph;
//import edu.cmu.tetrad.graph.Graph;
//import edu.cmu.tetrad.graph.IndependenceFact;
//import edu.cmu.tetrad.graph.Node;
//import edu.cmu.tetrad.util.RandomUtil;
//import edu.cmu.tetrad.util.TetradMatrix;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.Collections;
//import java.util.HashMap;
//import java.util.HashSet;
//import java.util.Iterator;
//import java.util.List;
//import java.util.Map;
//import java.util.Set;
//
///**
// * Uses BCInference by Cooper and Bui to calculate probabilistic conditional independence judgments.
// *
// * @author Fattaneh Jabbari 9/2019
// */
//public class IndTestProbabilisticISBDeu implements IndependenceTest {
//
//	private boolean threshold = false;
//
//	/**
//	 * The data set for which conditional  independence judgments are requested.
//	 */
//	private final DataSet data;
//	private final DataSet test;
//	private DataSet data_is;
//	private DataSet data_res;
//	private final int[][] data_array;
//	private final int[][] test_array;
//
//	/**
//	 * The nodes of the data set.
//	 */
//	private List<Node> nodes;
//
//	private final int[] nodeDimensions ;
//
//	/**
//	 * Indices of the nodes.
//	 */
//	private Map<Node, Integer> indices;
//
//	/**
//	 * A map from independence facts to their probabilities of independence.
//	 */
//	private Map<IndependenceFact, Double> H;
//
//	private Map<IndependenceFact, Double> H_population;
//	private Graph populationGraph;
//	private IndependenceTest populationDsep;
//	private double posterior;
//	private boolean verbose = false;
//
//	private double cutoff = 0.5;
//
//	//==========================CONSTRUCTORS=============================//
//	/**
//	 * Initializes the test using a discrete data sets.
//	 */
//	public IndTestProbabilisticISBDeu(DataSet dataSet, DataSet test, Map<IndependenceFact, Double> H_population, Graph populationGraph) {
//		if (!dataSet.isDiscrete()) {
//			throw new IllegalArgumentException("Not a discrete data set.");
//
//		}
//
//		this.H_population = H_population;
//		this.populationGraph = populationGraph;
//		this.data = dataSet;
//		this.test = test;
//
//		// dsep test for population graph
//		this.populationDsep = new IndTestDSep(this.populationGraph);
//
//		//  convert the data and the test case to an array
//		this.test_array = new int[this.test.getNumRows()][this.test.getNumColumns()];
//		for (int i = 0; i < test.getNumRows(); i++) {
//			for (int j = 0; j < test.getNumColumns(); j++) {
//				this.test_array[i][j] = test.getInt(i, j);
//			}
//		}
//
//		this.data_array = new int[dataSet.getNumRows()][dataSet.getNumColumns()];
//
//		for (int i = 0; i < dataSet.getNumRows(); i++) {
//			for (int j = 0; j < dataSet.getNumColumns(); j++) {
//				this.data_array[i][j] = dataSet.getInt(i, j);
//			}
//		}
//
//
//		this.nodeDimensions = new int[dataSet.getNumColumns() + 2];
//
//		for (int j = 0; j < dataSet.getNumColumns(); j++) {
//			DiscreteVariable variable = (DiscreteVariable) (dataSet.getVariable(j));
//			int numCategories = variable.getNumCategories();
//			this.nodeDimensions[j + 1] = numCategories;
//		}
//
//		nodes = dataSet.getVariables();
//
//		indices = new HashMap<>();
//
//		for (int i = 0; i < nodes.size(); i++) {
//			indices.put(nodes.get(i), i);
//		}
//
//		this.H = new HashMap<>();
//	}
//
//	@Override
//	public IndependenceTest indTestSubset(List<Node> vars) {
//		throw new UnsupportedOperationException();
//	}
//
//	@Override
//	public boolean isIndependent(Node x, Node y, List<Node> z) {
//		Node[] _z = z.toArray(new Node[z.size()]);
//		return isIndependent(x, y, _z);
//	}
//
//	@Override
//	public boolean isIndependent(Node x, Node y, Node... z) {
//		IndependenceFact key = new IndependenceFact(x, y, z);
//
//
//		double pInd = Double.NaN;      
//
//		if (!H.containsKey(key)) {
//
//			// convert set z to an array of indicies
//			int[] _z = new int[z.length];
//			for (int i = 0; i < z.length; i++) {
//				_z[i] = indices.get(z[i]);
//			}
//
//			if (_z.length == 0){
//				BDeuScorWOprior bic = new BDeuScorWOprior(this.data);
//				pInd = computeInd(bic, x, y, z);
//			}
//
//			else{
//				double pInd_is = Double.NaN;
//				double pTotalPopulation = Double.NaN;
//				boolean first = true;
//
//				// split the data based on array _z
//				splitData(_z);
//
//				// compute BSC based on D that matches values of _z in the test case
//				//				System.out.println("ind key: " + key);
//				if(this.data_is.getNumRows() > 0){ 
//					BDeuScorWOprior bic_is = new BDeuScorWOprior(this.data_is);
//					pInd_is = computeInd(bic_is, x, y, z);
//				}
//				else{
//					pInd_is = 0.5;
//				}
//
//				// compute BSC based on D that does not match values of _z in the test case
//				BDeuScorWOprior bic_res = new BDeuScorWOprior(this.data_res);
//				Map<IndependenceFact, Double> popConstraints = new HashMap<IndependenceFact, Double>();
//				Map<IndependenceFact, Double> popConstraints_old = new HashMap<IndependenceFact, Double>();
//
//				for (IndependenceFact k : this.H_population.keySet()){					
//					if ((k.getX().equals(x) && k.getY().equals(y)) ||(k.getX().equals(y) && k.getY().equals(x))){
//						//						System.out.println("k: " + k );
//						//						System.out.println("p_old = " + this.H_population.get(k));
//
//						// convert set z to an array of indecies
//						Node[] _kz = k.getZ().toArray(new Node[k.getZ().size()]);
//						double pXYPopulation =  computeInd(bic_res, k.getX(), k.getY(), _kz);
//
//						//						System.out.println("p_new = " + pXYPopulation);
//
//						popConstraints_old.put(k, this.H_population.get(k));
//						popConstraints.put(k, pXYPopulation);
//
//						if (this.populationDsep.isIndependent(k.getX(), k.getY(), k.getZ())){
//							if (first){
//								pTotalPopulation = Math.log10(pXYPopulation);
//								first = false;
//							}
//							else{
//								//							System.out.println("INDEP -- P_before" + pTotalPopulation);
//								pTotalPopulation += Math.log10(pXYPopulation);
//								//							System.out.println("INDEP -- P_after" + pTotalPopulation);
//							}
//
//						}
//						else{
//							if (first){								
//								pTotalPopulation = Math.log10(1 - pXYPopulation);
//								first = false;
//							}
//							else{
//								//							System.out.println("DEP -- P_before: " + pTotalPopulation);
//								pTotalPopulation += Math.log10(1 - pXYPopulation);
//								//							System.out.println("DEP -- P_after: " + pTotalPopulation);
//							}
//						}
//					}
//				}
//				if(popConstraints.size()==0){
//					pTotalPopulation = Math.log10(0.5);
//				}
//
//				//				System.out.println("popConstraints_old: " + popConstraints_old);
//				//				System.out.println("popConstraints:     " + popConstraints);
//				//				System.out.println("pInd_is: " + pInd_is);
//				//				System.out.println("pRes: " + Math.pow(10,pTotalPopulation));
//				pInd = pTotalPopulation + Math.log10(pInd_is);
//				pInd = Math.pow(10, pInd);
//			}
//
//			H.put(key, pInd);
//
//		}else {
//			pInd = H.get(key);
//		}
//
//		//        System.out.println("pInd_old: " + pInd_old);
//		//        System.out.println("pInd: " + pInd);
//		//        System.out.println("--------------------");
//		double p = pInd; 
//
//		this.posterior = p;
//
//		boolean ind ;
//		if (this.threshold){
//			ind = (p >= cutoff);
//		}
//		else{
//			ind = RandomUtil.getInstance().nextDouble() < p;
//		}
//
//		if (ind) {
//			return true;
//		} else {
//			return false;
//		}
//	}
//
//	private double computeInd(BDeuScorWOprior bic, Node x, Node y, Node... z) {
//		double pInd = Double.NaN;
//		List<Node> _z = new ArrayList<>();
//		_z.add(x);
//		_z.add(y);
//		Collections.addAll(_z, z);
//
//		Graph indBN = new EdgeListGraph(_z);
//		for (Node n : z){
//			indBN.addDirectedEdge(n, x);
//			indBN.addDirectedEdge(n, y);
//		}
//
//		Graph depBN = new EdgeListGraph(_z);
//		depBN.addDirectedEdge(x, y);
//		for (Node n : z){
//			depBN.addDirectedEdge(n, x);
//			depBN.addDirectedEdge(n, y);
//		}
//		double indPrior = Math.log(0.5);
//		double indScore = scoreDag(indBN,bic);
//		//      double indScore = scoreDag(indBN, bic, false, null, null);
//		double scoreIndAll = indScore + indPrior;
//
//
//		double depScore = scoreDag(depBN, bic);
//		//      double depScore = scoreDag(depBN, bic, true, x, y);
//		double depPrior = Math.log(1 - indPrior);
//		double scoreDepAll = depScore + depPrior;
//
//		double scoreAll = lnXpluslnY(scoreIndAll, scoreDepAll);
//		//  	System.out.println("scoreDepAll: " + scoreDepAll);
//		//      System.out.println("scoreIndAll: " + scoreIndAll);
//		//      System.out.println("scoreAll: " + scoreAll);
//
//		pInd = Math.exp(scoreIndAll - scoreAll);
//
//		return pInd;
//	}
//
//
//	//        double indPrior = Math.log(0.5);
//	//        double indScore = scoreDag(indBN, bic_is);
//	//        double scoreIndAll = indScore + indPrior;
//	//
//	//        
//	//        double depScore = scoreDag(depBN, bic_is);
//	//        double depPrior = Math.log(1 - indPrior);
//	//        double scoreDepAll = depScore + depPrior;
//	//
//	//        double scoreAll = lnXpluslnY(scoreIndAll, scoreDepAll);
//	////    	System.out.println("scoreDepAll: " + scoreDepAll);
//	////        System.out.println("scoreIndAll: " + scoreIndAll);
//	////        System.out.println("scoreAll: " + scoreAll);
//	//
//	//        pInd = Math.exp(scoreIndAll - scoreAll);
//	//        
//	//        return pInd;
//	//	}
//
//	private void splitData(int[] parents){
//
//		int sampleSize = this.data.getNumRows();
//		int numVariables = this.data.getNumColumns();
//		ArrayList<Integer> rows_is = new ArrayList<>();
//		ArrayList<Integer> rows_res = new ArrayList<>();
//
//		for (int i = 0; i < sampleSize; i++){
//			int[] parentValuesTest = new int[parents.length];
//			int[] parentValuesCase = new int[parents.length];
//
//			for (int p = 0; p < parents.length ; p++){
//				parentValuesTest[p] =  test_array[0][parents[p]];
//				parentValuesCase[p] = data_array[i][parents[p]];
//			}
//			int [] row = new int[numVariables];
//			for (int j = 0; j < numVariables; j++){				
//				row[j] = data_array[i][j];
//			}
//			if (Arrays.equals(parentValuesCase, parentValuesTest)){
//				rows_is.add(i);
//			}
//			else{
//				rows_res.add(i);
//			}		
//		}
//
//		//		this.data_is = new BoxDataSet(((BoxDataSet) this.data).getDataBox(), this.data.getVariables()); 
//		this.data_is = new BoxDataSet((BoxDataSet)this.data);
//
//		//		this.data_res = new BoxDataSet(((BoxDataSet) this.data).getDataBox(), this.data.getVariables());
//		this.data_res = new BoxDataSet((BoxDataSet)this.data);
//
//		this.data_is.removeRows(rows_res.stream().mapToInt(i -> i).toArray());
//		this.data_res.removeRows(rows_is.stream().mapToInt(i -> i).toArray());
//		//		System.out.println("data     :" + this.data.getNumRows());
//		//		System.out.println("data is  :" + this.data_is.getNumRows());
//		//		System.out.println("data res :" + this.data_res.getNumRows());
//	}
//
//	public Map<IndependenceFact, Double> groupbyXY(Map<IndependenceFact, Double> H, Node x, Node y){
//		Map<IndependenceFact, Double> H_xy = new HashMap<IndependenceFact, Double>();
//		for (IndependenceFact k : H.keySet()){					
//			if ((k.getX().equals(x) && k.getY().equals(y) && k.getZ().size() > 0) ||(k.getX().equals(y) && k.getY().equals(x) && k.getZ().size() > 0)){
//				H_xy.put(k, H.get(k));
//			}
//		}
//		return H_xy;
//	}
//	
//	public void splitDatabyXY(DataSet data, DataSet data_xy, DataSet data_rest, Map<IndependenceFact, Double> H_xy){
//
//		int sampleSize = data.getNumRows();
//		int numVariables = data.getNumColumns();
//
//		Set<Integer> rows_is = new HashSet<>();
//		Set<Integer> rows_res = new HashSet<>();
//		for (int i = 0; i < data.getNumRows(); i++){
//			rows_res.add(i);
//		}
//		
//		Node x = null, y = null;
//		for(IndependenceFact f : H_xy.keySet()){
////			System.out.println("f: " + f);
//			x = f.getX();
//			y = f.getY();
//			Node[] z = f.getZ().toArray(new Node[f.getZ().size()]);
//			int[] parents = new int[z.length];
//			for (int i = 0; i < z.length; i++) {
//				parents[i] = indices.get(z[i]);
//			}
//			
//			for (int i = 0; i < sampleSize; i++){
//				int[] parentValuesTest = new int[parents.length];
//				int[] parentValuesCase = new int[parents.length];
//
//				for (int p = 0; p < parents.length ; p++){
//					parentValuesTest[p] =  test_array[0][parents[p]];
//					parentValuesCase[p] = data_array[i][parents[p]];
//				}
//				
//				if (Arrays.equals(parentValuesCase, parentValuesTest)){
//					rows_is.add(i);
//					rows_res.remove(i);
//				}		
//			}
////			System.out.println("rows_is: " + rows_is);
////			System.out.println("rows_res: " + rows_res);
//		}
//
//		data_xy.removeRows(rows_res.stream().mapToInt(i -> i).toArray());
//		data_rest.removeRows(rows_is.stream().mapToInt(i -> i).toArray());
////		System.out.println("data_xy: " + data_xy.getNumRows());
////		System.out.println("data_rest: " + data_rest.getNumRows());
//
//	}
//
//	public double scoreDag(Graph dag, BDeuScorWOprior bic_is) {
//
//		double _score = 0.0;
//
//		for (Node y : dag.getNodes()) {
//			Set<Node> parents = new HashSet<>(dag.getParents(y));
//			int parentIndices[] = new int[parents.size()];
//			Iterator<Node> pi = parents.iterator();
//			int count = 0;
//
//			while (pi.hasNext()) {
//				Node nextParent = pi.next();
//				parentIndices[count++] = this.indices.get(nextParent);
//			}
//
//			int yIndex = this.indices.get(y);
//			_score += bic_is.localScore(yIndex, parentIndices);
//		}
//
//		return _score;
//	}
//
//	/**
//	 * Takes ln(x) and ln(y) as input, and returns ln(x + y)
//	 *
//	 * @param lnX is natural log of x
//	 * @param lnY is natural log of y
//	 * @return natural log of x plus y
//	 */
//	private static final int MININUM_EXPONENT = -1022;
//	protected double lnXpluslnY(double lnX, double lnY) {
//		double lnYminusLnX, temp;
//
//		if (lnY > lnX) {
//			temp = lnX;
//			lnX = lnY;
//			lnY = temp;
//		}
//
//		lnYminusLnX = lnY - lnX;
//
//		if (lnYminusLnX < MININUM_EXPONENT) {
//			return lnX;
//		} else {
//			return Math.log1p(Math.exp(lnYminusLnX)) + lnX;
//		}
//	}
//
//	//    public double probConstraint(BCInference.OP op, Node x, Node y, Node[] z) {
//	//
//	//        int _x = indices.get(x) + 1;
//	//        int _y = indices.get(y) + 1;
//	//
//	//        int[] _z = new int[z.length + 1];
//	//        _z[0] = z.length;
//	//        for (int i = 0; i < z.length; i++) {
//	//            _z[i + 1] = indices.get(z[i]) + 1;
//	//        }
//	//
//	//        return bci.probConstraint(op, _x, _y, _z);
//	//    }
//
//	@Override
//	public boolean isDependent(Node x, Node y, List<Node> z) {
//		Node[] _z = z.toArray(new Node[z.size()]);
//		return !isIndependent(x, y, _z);
//	}
//
//	@Override
//	public boolean isDependent(Node x, Node y, Node... z) {
//		return !isIndependent(x, y, z);
//	}
//
//	@Override
//	public double getPValue() {
//		return posterior;
//	}
//
//	@Override
//	public List<Node> getVariables() {
//		return nodes;
//	}
//
//	@Override
//	public Node getVariable(String name) {
//		for (Node node : nodes) {
//			if (name.equals(node.getName())) return node;
//		}
//
//		return null;
//	}
//
//	@Override
//	public List<String> getVariableNames() {
//		List<String> names = new ArrayList<>();
//
//		for (Node node : nodes) {
//			names.add(node.getName());
//		}
//		return names;
//	}
//
//	@Override
//	public boolean determines(List<Node> z, Node y) {
//		throw new UnsupportedOperationException();
//	}
//
//	@Override
//	public double getAlpha() {
//		throw new UnsupportedOperationException();
//	}
//
//	@Override
//	public void setAlpha(double alpha) {
//		throw new UnsupportedOperationException();
//	}
//
//	@Override
//	public DataModel getData() {
//		return data;
//	}
//
//	@Override
//	public ICovarianceMatrix getCov() {
//		return null;
//	}
//
//	@Override
//	public List<DataSet> getDataSets() {
//		return null;
//	}
//
//	@Override
//	public int getSampleSize() {
//		return 0;
//	}
//
//	@Override
//	public List<TetradMatrix> getCovMatrices() {
//		return null;
//	}
//
//	@Override
//	public double getScore() {
//		return getPValue();
//	}
//
//	public Map<IndependenceFact, Double> getH() {
//		return new HashMap<>(H);
//	}
//
//	//    private double probOp(BCInference.OP type, double pInd) {
//	//        double probOp;
//	//
//	//        if (BCInference.OP.independent == type) {
//	//            probOp = pInd;
//	//        } else {
//	//            probOp = 1.0 - pInd;
//	//        }
//	//
//	//        return probOp;
//	//    }
//
//	public double getPosterior() {
//		return posterior;
//	}
//
//	@Override
//	public boolean isVerbose() {
//		return verbose;
//	}
//
//	@Override
//	public void setVerbose(boolean verbose) {
//		this.verbose = verbose;
//	}
//
//	/**
//	 * @param noRandomizedGeneratingConstraints
//	 */
//	public void setThreshold(boolean noRandomizedGeneratingConstraints) {
//		this.threshold = noRandomizedGeneratingConstraints;
//	}
//
//	public void setCutoff(double cutoff) {
//		this.cutoff = cutoff;
//	}
//}
//
//
//


///////////////////////////////////////////////////////////////////////////////
//// For information as to what this class does, see the Javadoc, below.       //
//// Copyright (C) 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,       //
//// 2007, 2008, 2009, 2010, 2014, 2015 by Peter Spirtes, Richard Scheines, Joseph   //
//// Ramsey, and Clark Glymour.                                                //
////                                                                           //
//// This program is free software; you can redistribute it and/or modify      //
//// it under the terms of the GNU General Public License as published by      //
//// the Free Software Foundation; either version 2 of the License, or         //
//// (at your option) any later version.                                       //
////                                                                           //
//// This program is distributed in the hope that it will be useful,           //
//// but WITHOUT ANY WARRANTY; without even the implied warranty of            //
//// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
//// GNU General Public License for more details.                              //
////                                                                           //
//// You should have received a copy of the GNU General Public License         //
//// along with this program; if not, write to the Free Software               //
//// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA //
/////////////////////////////////////////////////////////////////////////////////
//
//package edu.cmu.tetrad.search;
//
//import edu.cmu.tetrad.data.BoxDataSet;
//import edu.cmu.tetrad.data.DataModel;
//import edu.cmu.tetrad.data.DataSet;
//import edu.cmu.tetrad.data.DiscreteVariable;
//import edu.cmu.tetrad.data.ICovarianceMatrix;
//import edu.cmu.tetrad.graph.EdgeListGraph;
//import edu.cmu.tetrad.graph.Graph;
//import edu.cmu.tetrad.graph.IndependenceFact;
//import edu.cmu.tetrad.graph.Node;
//import edu.cmu.tetrad.util.RandomUtil;
//import edu.cmu.tetrad.util.TetradMatrix;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.Collections;
//import java.util.HashMap;
//import java.util.HashSet;
//import java.util.Iterator;
//import java.util.List;
//import java.util.Map;
//import java.util.Set;
//
///**
// * Uses BCInference by Cooper and Bui to calculate probabilistic conditional independence judgments.
// *
// * @author Fattaneh Jabbari 9/2019
// */
//public class IndTestProbabilisticISBDeu implements IndependenceTest {
//
//	private boolean threshold = false;
//
//	/**
//	 * The data set for which conditional  independence judgments are requested.
//	 */
//	private final DataSet data;
//	private final DataSet test;
//	private final int[][] data_array;
//	private final int[][] test_array;
//
//	/**
//	 * The nodes of the data set.
//	 */
//	private List<Node> nodes;
//
//	private final int[] nodeDimensions ;
//
//	/**
//	 * Indices of the nodes.
//	 */
//	private Map<Node, Integer> indices;
//
//	/**
//	 * A map from independence facts to their probabilities of independence.
//	 */
//	private Map<IndependenceFact, Double> H;
//
//	private Map<IndependenceFact, Double> H_population;
//	private Graph populationGraph;
//	private IndependenceTest populationDsep;
//	private double posterior;
//	private boolean verbose = false;
//
//	private double cutoff = 0.5;
//
//	//==========================CONSTRUCTORS=============================//
//	/**
//	 * Initializes the test using a discrete data sets.
//	 */
//	public IndTestProbabilisticISBDeu(DataSet dataSet, DataSet test, Map<IndependenceFact, Double> H_population, Graph populationGraph) {
//		if (!dataSet.isDiscrete()) {
//			throw new IllegalArgumentException("Not a discrete data set.");
//
//		}
//
//		this.H_population = H_population;
//		this.populationGraph = populationGraph;
//		this.data = dataSet;
//		this.test = test;
//
//		// dsep test for population graph
//		this.populationDsep = new IndTestDSep(this.populationGraph);
//
//		//  convert the data and the test case to an array
//		this.test_array = new int[this.test.getNumRows()][this.test.getNumColumns()];
//		for (int i = 0; i < test.getNumRows(); i++) {
//			for (int j = 0; j < test.getNumColumns(); j++) {
//				this.test_array[i][j] = test.getInt(i, j);
//			}
//		}
//
//		this.data_array = new int[dataSet.getNumRows()][dataSet.getNumColumns()];
//
//		for (int i = 0; i < dataSet.getNumRows(); i++) {
//			for (int j = 0; j < dataSet.getNumColumns(); j++) {
//				this.data_array[i][j] = dataSet.getInt(i, j);
//			}
//		}
//
//
//		this.nodeDimensions = new int[dataSet.getNumColumns() + 2];
//
//		for (int j = 0; j < dataSet.getNumColumns(); j++) {
//			DiscreteVariable variable = (DiscreteVariable) (dataSet.getVariable(j));
//			int numCategories = variable.getNumCategories();
//			this.nodeDimensions[j + 1] = numCategories;
//		}
//
//		nodes = dataSet.getVariables();
//
//		indices = new HashMap<>();
//
//		for (int i = 0; i < nodes.size(); i++) {
//			indices.put(nodes.get(i), i);
//		}
//
//		this.H = new HashMap<>();
//	}
//
//	@Override
//	public IndependenceTest indTestSubset(List<Node> vars) {
//		throw new UnsupportedOperationException();
//	}
//
//	@Override
//	public boolean isIndependent(Node x, Node y, List<Node> z) {
//		Node[] _z = z.toArray(new Node[z.size()]);
//		return isIndependent(x, y, _z);
//	}
//
//	@Override
//	public boolean isIndependent(Node x, Node y, Node... z) {
//		IndependenceFact key = new IndependenceFact(x, y, z);
//
//
//		double pInd = Double.NaN;      
//
//		if (!H.containsKey(key)) {
//
//			// convert set z to an array of indicies
//			int[] _z = new int[z.length];
//			for (int i = 0; i < z.length; i++) {
//				_z[i] = indices.get(z[i]);
//			}
//
//			if (_z.length == 0){
//				BDeuScoreWOprior bic = new BDeuScoreWOprior(this.data);
//				pInd = computeInd(bic, x, y, z);
////				pInd = computeIndWithMultipleStructures(bic, x, y, z);
//			}
//
//			else{
//				double pInd_is = Double.NaN;
//				double pTotalPopulation = 0.0;
////				boolean first = true;
//
//				// split the data based on array _z
//				DataSet data_is = new BoxDataSet((BoxDataSet) this.data);
//				DataSet data_rest = new BoxDataSet((BoxDataSet) this.data);
//
//				splitData(data_is, data_rest, _z);
//
//				// compute BSC based on D that matches values of _z in the test case
//				if(data_is.getNumRows() > 0){ 
//					BDeuScoreWOprior bic_is = new BDeuScoreWOprior(data_is);
////					pInd_is = computeIndWithMultipleStructures(bic_is, x, y, z);
//					pInd_is = computeInd(bic_is, x, y, z);
//				}
//				else{
//					pInd_is = 0.5;
//				}
//
//				// compute BSC based on D that does not match values of _z in the test case
//				BDeuScoreWOprior bic_res = new BDeuScoreWOprior(data_rest);
//				Map<IndependenceFact, Double> popConstraints = new HashMap<IndependenceFact, Double>();
//				Map<IndependenceFact, Double> popConstraints_old = new HashMap<IndependenceFact, Double>();
//
//				for (IndependenceFact k : this.H_population.keySet()){					
//					if ((k.getX().equals(x) && k.getY().equals(y)) ||(k.getX().equals(y) && k.getY().equals(x))){
//						
//						// convert set z to an array of indecies
//						Node[] _kz = k.getZ().toArray(new Node[k.getZ().size()]);
//						double pXYPopulation =  computeInd(bic_res, k.getX(), k.getY(), _kz);
////						double pXYPopulation =  computeIndWithMultipleStructures(bic_res, k.getX(), k.getY(), _kz);
//
//
//						popConstraints_old.put(k, this.H_population.get(k));
//						popConstraints.put(k, pXYPopulation);
//
//						if (this.populationDsep.isIndependent(k.getX(), k.getY(), k.getZ())){
////							if (first){
////								pTotalPopulation = Math.log10(pXYPopulation);
////								first = false;
////							}
////							else{
//							pTotalPopulation += Math.log10(pXYPopulation);
////							}
//
//						}
//						else{
////							if (first){								
////								pTotalPopulation = Math.log10(1 - pXYPopulation);
////								first = false;
////							}
////							else{
//								//							System.out.println("DEP -- P_before: " + pTotalPopulation);
//							pTotalPopulation += Math.log10(1 - pXYPopulation);
//								//							System.out.println("DEP -- P_after: " + pTotalPopulation);
////							}
//						}
//					}
//				}
//				if(popConstraints.size()==0){
//					pTotalPopulation = Math.log10(0.5);
//				}
//
//				//				System.out.println("popConstraints_old: " + popConstraints_old);
//				//				System.out.println("popConstraints:     " + popConstraints);
//				//				System.out.println("pInd_is: " + pInd_is);
//				//				System.out.println("pRes: " + Math.pow(10,pTotalPopulation));
//				pInd = pTotalPopulation + Math.log10(pInd_is);
//				pInd = Math.pow(10, pInd);
//			}
//
//			H.put(key, pInd);
//
//		}else {
//			pInd = H.get(key);
//		}
//
//		//        System.out.println("pInd_old: " + pInd_old);
//		//        System.out.println("pInd: " + pInd);
//		//        System.out.println("--------------------");
//		double p = pInd; 
//
//		this.posterior = p;
//
//		boolean ind ;
//		if (this.threshold){
//			ind = (p >= cutoff);
//		}
//		else{
//			ind = RandomUtil.getInstance().nextDouble() < p;
//		}
//
//		if (ind) {
//			return true;
//		} else {
//			return false;
//		}
//	}
//
//	public double computeIndIS(Node x, Node y, Node... z) {
//		double pInd = Double.NaN;      		
//		// convert set z to an array of indecies
//		int[] _z = new int[z.length];
//		for (int i = 0; i < z.length; i++) {
//			_z[i] = indices.get(z[i]);
//		}
//
//		if (_z.length == 0){
//			System.out.println("ERRORRRRR   Z= 0");
//		}
//		else{
//			// split the data based on array _z
//			DataSet data_is = new BoxDataSet((BoxDataSet) this.data);
//			DataSet data_rest = new BoxDataSet((BoxDataSet) this.data);
//			splitData(data_is, data_rest, _z);
//			
//			// compute BSC based on D that matches values of _z in the test case
//			if(data_is.getNumRows() > 0){ 
//				BDeuScoreWOprior bic_is = new BDeuScoreWOprior(data_is);
//				pInd = computeInd(bic_is, x, y, z);
//			}
//			else{
//				pInd = 0.5;
//			}
////			System.out.println("pInd_is: " + pInd_is);
//
//		}
//
//		double p = pInd; 
//
//		this.posterior = p;
//
//		return p;
//	}
//
//
//	private double computeInd(BDeuScoreWOprior bic, Node x, Node y, Node... z) {
//		double pInd = Double.NaN;
//		List<Node> _z = new ArrayList<>();
//		_z.add(x);
//		_z.add(y);
//		Collections.addAll(_z, z);
//
//		Graph indBN = new EdgeListGraph(_z);
//		for (Node n : z){
//			indBN.addDirectedEdge(n, x);
//			indBN.addDirectedEdge(n, y);
//		}
//
//		Graph depBN = new EdgeListGraph(_z);
//		depBN.addDirectedEdge(x, y);
//		for (Node n : z){
//			depBN.addDirectedEdge(n, x);
//			depBN.addDirectedEdge(n, y);
//		}
//		double indPrior = Math.log(0.5);
//		double indScore = scoreDag(indBN,bic);
//		//      double indScore = scoreDag(indBN, bic, false, null, null);
//		double scoreIndAll = indScore + indPrior;
//
//
//		double depScore = scoreDag(depBN, bic);
//		//      double depScore = scoreDag(depBN, bic, true, x, y);
//		double depPrior = Math.log(1 - indPrior);
//		double scoreDepAll = depScore + depPrior;
//
//		double scoreAll = lnXpluslnY(scoreIndAll, scoreDepAll);
//		//  	System.out.println("scoreDepAll: " + scoreDepAll);
//		//      System.out.println("scoreIndAll: " + scoreIndAll);
//		//      System.out.println("scoreAll: " + scoreAll);
//
//		pInd = Math.exp(scoreIndAll - scoreAll);
//
//		return pInd;
//	}
//	
//	private double computeIndWithMultipleStructures(BDeuScoreWOprior bic, Node x, Node y, Node... z) {
//		double pInd = Double.NaN;
//		List<Node> allNodes = new ArrayList<>();
//        allNodes.add(x);
//        allNodes.add(y);
//        Collections.addAll(allNodes, z);
//        Graph[] indBNs = new Graph[4];
//        for (int i = 0; i < indBNs.length; i++){
//        	indBNs[i] = new EdgeListGraph(allNodes);
//        }
//        
//        for (Node n : z){
//        	indBNs[1].addDirectedEdge(n, x);
//        	indBNs[1].addDirectedEdge(n, y);
//        	indBNs[2].addDirectedEdge(n, x);
//        	indBNs[3].addDirectedEdge(n, y);
//        }
//        
//        Graph[] depBNs = new Graph[7];
//        for (int i = 0; i < depBNs.length; i++){
//        	depBNs[i] = new EdgeListGraph(allNodes);
//        }
//        
//        depBNs[0].addDirectedEdge(x, y);
//        depBNs[1].addDirectedEdge(x, y);
//        depBNs[2].addDirectedEdge(y, x);
//        depBNs[4].addDirectedEdge(x, y);
//        depBNs[5].addDirectedEdge(y, x);
//    	depBNs[6].addDirectedEdge(x, y);
//
//        for (Node n : z){
//        	depBNs[1].addDirectedEdge(n, x);
//        	depBNs[2].addDirectedEdge(n, y);
//        	depBNs[3].addDirectedEdge(x, n);
//        	depBNs[3].addDirectedEdge(y, n);
//        	depBNs[4].addDirectedEdge(n, x);
//        	depBNs[4].addDirectedEdge(n, y);
//        	depBNs[5].addDirectedEdge(n, x);
//        	depBNs[6].addDirectedEdge(n, y);
//        }
//        
//        double scoreIndAll = Double.NEGATIVE_INFINITY; 
//        double indPrior = Math.log(0.5 / (indBNs.length));
//        double[] indScores = new double[indBNs.length];
//        double[] depScores = new double[depBNs.length];
//    	
//        for (int i = 0; i < indScores.length; i++){
//        	indScores[i] = scoreDag(indBNs[i], bic);
//        	scoreIndAll = lnXpluslnY(scoreIndAll, (indScores[i]));// + indPrior));
//        }
//        
//        
//        double scoreDepAll = Double.NEGATIVE_INFINITY;
//        double depPrior = Math.log(0.5 / (depBNs.length));
//        for (int i = 0; i < depScores.length; i++){
//        	depScores[i] = scoreDag(depBNs[i], bic);
//        	scoreDepAll = lnXpluslnY(scoreDepAll, (depScores[i]));// + depPrior));
//        }
//
//        double scoreAll = lnXpluslnY(scoreIndAll, scoreDepAll);
//
//        pInd = Math.exp(scoreIndAll - scoreAll);
//        
//        return pInd;
//	}
//
//
//	//        double indPrior = Math.log(0.5);
//	//        double indScore = scoreDag(indBN, bic_is);
//	//        double scoreIndAll = indScore + indPrior;
//	//
//	//        
//	//        double depScore = scoreDag(depBN, bic_is);
//	//        double depPrior = Math.log(1 - indPrior);
//	//        double scoreDepAll = depScore + depPrior;
//	//
//	//        double scoreAll = lnXpluslnY(scoreIndAll, scoreDepAll);
//	////    	System.out.println("scoreDepAll: " + scoreDepAll);
//	////        System.out.println("scoreIndAll: " + scoreIndAll);
//	////        System.out.println("scoreAll: " + scoreAll);
//	//
//	//        pInd = Math.exp(scoreIndAll - scoreAll);
//	//        
//	//        return pInd;
//	//	}
//	private void splitData(DataSet data_xy, DataSet data_rest, int[] parents){
//		int sampleSize = data.getNumRows();
//
//		Set<Integer> rows_is = new HashSet<>();
//		Set<Integer> rows_res = new HashSet<>();
//		
//		for (int i = 0; i < data.getNumRows(); i++){
//			rows_res.add(i);
//		}
//
////		for(IndependenceFact f : H_xy.keySet()){
//
//			for (int i = 0; i < sampleSize; i++){
//				int[] parentValuesTest = new int[parents.length];
//				int[] parentValuesCase = new int[parents.length];
//
//				for (int p = 0; p < parents.length ; p++){
//					parentValuesTest[p] =  test_array[0][parents[p]];
//					parentValuesCase[p] = data_array[i][parents[p]];
//				}
//
//				if (Arrays.equals(parentValuesCase, parentValuesTest)){
//					rows_is.add(i);
//					rows_res.remove(i);
//				}		
//			}
////		}
//
//		int[] is_array = new int[rows_is.size()];
//		int c = 0;
//		for(int row : rows_is) is_array[c++] = row;
//		
//		int[] res_array = new int[rows_res.size()];
//		c = 0;
//		for(int row : rows_res) res_array[c++] = row;
//		
//		Arrays.sort(is_array);
//		Arrays.sort(res_array);
//		
//		data_xy.removeRows(res_array);
//		data_rest.removeRows(is_array);
//		//		System.out.println("data_xy: " + data_xy.getNumRows());
//		//		System.out.println("data_rest: " + data_rest.getNumRows());
//
//	}
//
////	private void splitData(int[] parents){
////
////		int sampleSize = this.data.getNumRows();
////		int numVariables = this.data.getNumColumns();
//////		ArrayList<Integer> rows_is = new ArrayList<>();
//////		ArrayList<Integer> rows_res = new ArrayList<>();
////		Set<Integer> rows_is = new HashSet<>();
////		Set<Integer> rows_res = new HashSet<>();
////		for (int i = 0; i < data.getNumRows(); i++){
////			rows_res.add(i);
////		}
////
////		for (int i = 0; i < sampleSize; i++){
////			int[] parentValuesTest = new int[parents.length];
////			int[] parentValuesCase = new int[parents.length];
////
////			for (int p = 0; p < parents.length ; p++){
////				parentValuesTest[p] =  test_array[0][parents[p]];
////				parentValuesCase[p] = data_array[i][parents[p]];
////			}
////			int [] row = new int[numVariables];
////			for (int j = 0; j < numVariables; j++){				
////				row[j] = data_array[i][j];
////			}
////			if (Arrays.equals(parentValuesCase, parentValuesTest)){
////				rows_is.add(i);
////				rows_res.remove(i);
////			}		
////		}
////
////		//		this.data_is = new BoxDataSet(((BoxDataSet) this.data).getDataBox(), this.data.getVariables()); 
////		this.data_is = new BoxDataSet((BoxDataSet)this.data);
////
////		//		this.data_res = new BoxDataSet(((BoxDataSet) this.data).getDataBox(), this.data.getVariables());
////		this.data_res = new BoxDataSet((BoxDataSet)this.data);
////		
//////		System.out.println("is :" + rows_is);
//////		System.out.println("res :" + rows_res);
////		int[] is_array = new int[rows_is.size()];
////		int c = 0;
////		for(int row : rows_is) is_array[c++] = row;
////		int[] res_array = new int[rows_res.size()];
////		c = 0;
////		for(int row : rows_res) res_array[c++] = row;
////		Arrays.sort(is_array);
////		Arrays.sort(res_array);
////		this.data_is.removeRows(res_array);
////		this.data_res.removeRows(is_array);
////		//		System.out.println("data     :" + this.data.getNumRows());
////		//		System.out.println("data is  :" + this.data_is.getNumRows());
////		//		System.out.println("data res :" + this.data_res.getNumRows());
////	}
//
//	public Map<IndependenceFact, Double> groupbyXYI(Map<IndependenceFact, Double> H, Node x, Node y){
//		Map<IndependenceFact, Double> H_xy = new HashMap<IndependenceFact, Double>();
//		for (IndependenceFact k : H.keySet()){					
//			if ((k.getX().equals(x) && k.getY().equals(y) && k.getZ().size() > 0) ||(k.getX().equals(y) && k.getY().equals(x) && k.getZ().size() > 0)){
//				H_xy.put(k, H.get(k));
//			}
//		}
//		return H_xy;
//	}
//	public Map<IndependenceFact, Double> groupbyXYP(Map<IndependenceFact, Double> H, Node x, Node y){
//		Map<IndependenceFact, Double> H_xy = new HashMap<IndependenceFact, Double>();
//		for (IndependenceFact k : H.keySet()){					
//			if ((k.getX().equals(x) && k.getY().equals(y)) ||(k.getX().equals(y) && k.getY().equals(x))){
//				H_xy.put(k, H.get(k));
//			}
//		}
//		return H_xy;
//	}
//
//	public void splitDatabyXY(DataSet data, DataSet data_xy, DataSet data_rest, Map<IndependenceFact, Double> H_xy){
//
//		int sampleSize = data.getNumRows();
//
//		Set<Integer> rows_is = new HashSet<>();
//		Set<Integer> rows_res = new HashSet<>();
//		for (int i = 0; i < data.getNumRows(); i++){
//			rows_res.add(i);
//		}
//
//		for(IndependenceFact f : H_xy.keySet()){
//			Node[] z = f.getZ().toArray(new Node[f.getZ().size()]);
//			int[] parents = new int[z.length];
//			for (int i = 0; i < z.length; i++) {
//				parents[i] = indices.get(z[i]);
//			}
//
//			for (int i = 0; i < sampleSize; i++){
//				int[] parentValuesTest = new int[parents.length];
//				int[] parentValuesCase = new int[parents.length];
//
//				for (int p = 0; p < parents.length ; p++){
//					parentValuesTest[p] =  test_array[0][parents[p]];
//					parentValuesCase[p] = data_array[i][parents[p]];
//				}
//
//				if (Arrays.equals(parentValuesCase, parentValuesTest)){
//					rows_is.add(i);
//					rows_res.remove(i);
//				}		
//			}
//		}
//
//		int[] is_array = new int[rows_is.size()];
//		int c = 0;
//		for(int row : rows_is) is_array[c++] = row;
//		int[] res_array = new int[rows_res.size()];
//		c = 0;
//		for(int row : rows_res) res_array[c++] = row;
//		Arrays.sort(is_array);
//		Arrays.sort(res_array);
//		data_xy.removeRows(res_array);
//		data_rest.removeRows(is_array);
//		//		System.out.println("data_xy: " + data_xy.getNumRows());
//		//		System.out.println("data_rest: " + data_rest.getNumRows());
//
//	}
//
//	public double scoreDag(Graph dag, BDeuScoreWOprior bic_is) {
//
//		double _score = 0.0;
//
//		for (Node y : dag.getNodes()) {
//			Set<Node> parents = new HashSet<>(dag.getParents(y));
//			int parentIndices[] = new int[parents.size()];
//			Iterator<Node> pi = parents.iterator();
//			int count = 0;
//
//			while (pi.hasNext()) {
//				Node nextParent = pi.next();
//				parentIndices[count++] = this.indices.get(nextParent);
//			}
//
//			int yIndex = this.indices.get(y);
//			_score += bic_is.localScore(yIndex, parentIndices);
//		}
//
//		return _score;
//	}
//
//	/**
//	 * Takes ln(x) and ln(y) as input, and returns ln(x + y)
//	 *
//	 * @param lnX is natural log of x
//	 * @param lnY is natural log of y
//	 * @return natural log of x plus y
//	 */
//	private static final int MININUM_EXPONENT = -1022;
//	protected double lnXpluslnY(double lnX, double lnY) {
//		double lnYminusLnX, temp;
//
//		if (lnY > lnX) {
//			temp = lnX;
//			lnX = lnY;
//			lnY = temp;
//		}
//
//		lnYminusLnX = lnY - lnX;
//
//		if (lnYminusLnX < MININUM_EXPONENT) {
//			return lnX;
//		} else {
//			return Math.log1p(Math.exp(lnYminusLnX)) + lnX;
//		}
//	}
//
//	//    public double probConstraint(BCInference.OP op, Node x, Node y, Node[] z) {
//	//
//	//        int _x = indices.get(x) + 1;
//	//        int _y = indices.get(y) + 1;
//	//
//	//        int[] _z = new int[z.length + 1];
//	//        _z[0] = z.length;
//	//        for (int i = 0; i < z.length; i++) {
//	//            _z[i + 1] = indices.get(z[i]) + 1;
//	//        }
//	//
//	//        return bci.probConstraint(op, _x, _y, _z);
//	//    }
//
//	@Override
//	public boolean isDependent(Node x, Node y, List<Node> z) {
//		Node[] _z = z.toArray(new Node[z.size()]);
//		return !isIndependent(x, y, _z);
//	}
//
//	@Override
//	public boolean isDependent(Node x, Node y, Node... z) {
//		return !isIndependent(x, y, z);
//	}
//
//	@Override
//	public double getPValue() {
//		return posterior;
//	}
//
//	@Override
//	public List<Node> getVariables() {
//		return nodes;
//	}
//
//	@Override
//	public Node getVariable(String name) {
//		for (Node node : nodes) {
//			if (name.equals(node.getName())) return node;
//		}
//
//		return null;
//	}
//
//	@Override
//	public List<String> getVariableNames() {
//		List<String> names = new ArrayList<>();
//
//		for (Node node : nodes) {
//			names.add(node.getName());
//		}
//		return names;
//	}
//
//	@Override
//	public boolean determines(List<Node> z, Node y) {
//		throw new UnsupportedOperationException();
//	}
//
//	@Override
//	public double getAlpha() {
//		throw new UnsupportedOperationException();
//	}
//
//	@Override
//	public void setAlpha(double alpha) {
//		throw new UnsupportedOperationException();
//	}
//
//	@Override
//	public DataModel getData() {
//		return data;
//	}
//
//	@Override
//	public ICovarianceMatrix getCov() {
//		return null;
//	}
//
//	@Override
//	public List<DataSet> getDataSets() {
//		return null;
//	}
//
//	@Override
//	public int getSampleSize() {
//		return 0;
//	}
//
//	@Override
//	public List<TetradMatrix> getCovMatrices() {
//		return null;
//	}
//
//	@Override
//	public double getScore() {
//		return getPValue();
//	}
//
//	public Map<IndependenceFact, Double> getH() {
//		return new HashMap<>(H);
//	}
//
//	//    private double probOp(BCInference.OP type, double pInd) {
//	//        double probOp;
//	//
//	//        if (BCInference.OP.independent == type) {
//	//            probOp = pInd;
//	//        } else {
//	//            probOp = 1.0 - pInd;
//	//        }
//	//
//	//        return probOp;
//	//    }
//
//	public double getPosterior() {
//		return posterior;
//	}
//
//	@Override
//	public boolean isVerbose() {
//		return verbose;
//	}
//
//	@Override
//	public void setVerbose(boolean verbose) {
//		this.verbose = verbose;
//	}
//
//	/**
//	 * @param noRandomizedGeneratingConstraints
//	 */
//	public void setThreshold(boolean noRandomizedGeneratingConstraints) {
//		this.threshold = noRandomizedGeneratingConstraints;
//	}
//
//	public void setCutoff(double cutoff) {
//		this.cutoff = cutoff;
//	}
//}
//
//
//
//
///////////////////////////////////////////////////////////////////////////////////
////// For information as to what this class does, see the Javadoc, below.       //
////// Copyright (C) 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,       //
////// 2007, 2008, 2009, 2010, 2014, 2015 by Peter Spirtes, Richard Scheines, Joseph   //
////// Ramsey, and Clark Glymour.                                                //
//////                                                                           //
////// This program is free software; you can redistribute it and/or modify      //
////// it under the terms of the GNU General Public License as published by      //
////// the Free Software Foundation; either version 2 of the License, or         //
////// (at your option) any later version.                                       //
//////                                                                           //
////// This program is distributed in the hope that it will be useful,           //
////// but WITHOUT ANY WARRANTY; without even the implied warranty of            //
////// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
////// GNU General Public License for more details.                              //
//////                                                                           //
////// You should have received a copy of the GNU General Public License         //
////// along with this program; if not, write to the Free Software               //
////// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA //
///////////////////////////////////////////////////////////////////////////////////
////
////package edu.cmu.tetrad.search;
////
////import edu.cmu.tetrad.data.BoxDataSet;
////import edu.cmu.tetrad.data.DataModel;
////import edu.cmu.tetrad.data.DataSet;
////import edu.cmu.tetrad.data.DiscreteVariable;
////import edu.cmu.tetrad.data.ICovarianceMatrix;
////import edu.cmu.tetrad.graph.EdgeListGraph;
////import edu.cmu.tetrad.graph.Graph;
////import edu.cmu.tetrad.graph.IndependenceFact;
////import edu.cmu.tetrad.graph.Node;
////import edu.cmu.tetrad.util.RandomUtil;
////import edu.cmu.tetrad.util.TetradMatrix;
////import java.util.ArrayList;
////import java.util.Arrays;
////import java.util.Collections;
////import java.util.HashMap;
////import java.util.HashSet;
////import java.util.Iterator;
////import java.util.List;
////import java.util.Map;
////import java.util.Set;
////
/////**
//// * Uses BCInference by Cooper and Bui to calculate probabilistic conditional independence judgments.
//// *
//// * @author Fattaneh Jabbari 9/2019
//// */
////public class IndTestProbabilisticISBDeu implements IndependenceTest {
////
////	private boolean threshold = false;
////
////	/**
////	 * The data set for which conditional  independence judgments are requested.
////	 */
////	private final DataSet data;
////	private final DataSet test;
////	private DataSet data_is;
////	private DataSet data_res;
////	private final int[][] data_array;
////	private final int[][] test_array;
////
////	/**
////	 * The nodes of the data set.
////	 */
////	private List<Node> nodes;
////
////	private final int[] nodeDimensions ;
////
////	/**
////	 * Indices of the nodes.
////	 */
////	private Map<Node, Integer> indices;
////
////	/**
////	 * A map from independence facts to their probabilities of independence.
////	 */
////	private Map<IndependenceFact, Double> H;
////
////	private Map<IndependenceFact, Double> H_population;
////	private Graph populationGraph;
////	private IndependenceTest populationDsep;
////	private double posterior;
////	private boolean verbose = false;
////
////	private double cutoff = 0.5;
////
////	//==========================CONSTRUCTORS=============================//
////	/**
////	 * Initializes the test using a discrete data sets.
////	 */
////	public IndTestProbabilisticISBDeu(DataSet dataSet, DataSet test, Map<IndependenceFact, Double> H_population, Graph populationGraph) {
////		if (!dataSet.isDiscrete()) {
////			throw new IllegalArgumentException("Not a discrete data set.");
////
////		}
////
////		this.H_population = H_population;
////		this.populationGraph = populationGraph;
////		this.data = dataSet;
////		this.test = test;
////
////		// dsep test for population graph
////		this.populationDsep = new IndTestDSep(this.populationGraph);
////
////		//  convert the data and the test case to an array
////		this.test_array = new int[this.test.getNumRows()][this.test.getNumColumns()];
////		for (int i = 0; i < test.getNumRows(); i++) {
////			for (int j = 0; j < test.getNumColumns(); j++) {
////				this.test_array[i][j] = test.getInt(i, j);
////			}
////		}
////
////		this.data_array = new int[dataSet.getNumRows()][dataSet.getNumColumns()];
////
////		for (int i = 0; i < dataSet.getNumRows(); i++) {
////			for (int j = 0; j < dataSet.getNumColumns(); j++) {
////				this.data_array[i][j] = dataSet.getInt(i, j);
////			}
////		}
////
////
////		this.nodeDimensions = new int[dataSet.getNumColumns() + 2];
////
////		for (int j = 0; j < dataSet.getNumColumns(); j++) {
////			DiscreteVariable variable = (DiscreteVariable) (dataSet.getVariable(j));
////			int numCategories = variable.getNumCategories();
////			this.nodeDimensions[j + 1] = numCategories;
////		}
////
////		nodes = dataSet.getVariables();
////
////		indices = new HashMap<>();
////
////		for (int i = 0; i < nodes.size(); i++) {
////			indices.put(nodes.get(i), i);
////		}
////
////		this.H = new HashMap<>();
////	}
////
////	@Override
////	public IndependenceTest indTestSubset(List<Node> vars) {
////		throw new UnsupportedOperationException();
////	}
////
////	@Override
////	public boolean isIndependent(Node x, Node y, List<Node> z) {
////		Node[] _z = z.toArray(new Node[z.size()]);
////		return isIndependent(x, y, _z);
////	}
////
////	@Override
////	public boolean isIndependent(Node x, Node y, Node... z) {
////		IndependenceFact key = new IndependenceFact(x, y, z);
////
////
////		double pInd = Double.NaN;      
////
////		if (!H.containsKey(key)) {
////
////			// convert set z to an array of indicies
////			int[] _z = new int[z.length];
////			for (int i = 0; i < z.length; i++) {
////				_z[i] = indices.get(z[i]);
////			}
////
////			if (_z.length == 0){
////				BDeuScorWOprior bic = new BDeuScorWOprior(this.data);
////				pInd = computeInd(bic, x, y, z);
////			}
////
////			else{
////				double pInd_is = Double.NaN;
////				double pTotalPopulation = Double.NaN;
////				boolean first = true;
////
////				// split the data based on array _z
////				splitData(_z);
////
////				// compute BSC based on D that matches values of _z in the test case
////				//				System.out.println("ind key: " + key);
////				if(this.data_is.getNumRows() > 0){ 
////					BDeuScorWOprior bic_is = new BDeuScorWOprior(this.data_is);
////					pInd_is = computeInd(bic_is, x, y, z);
////				}
////				else{
////					pInd_is = 0.5;
////				}
////
////				// compute BSC based on D that does not match values of _z in the test case
////				BDeuScorWOprior bic_res = new BDeuScorWOprior(this.data_res);
////				Map<IndependenceFact, Double> popConstraints = new HashMap<IndependenceFact, Double>();
////				Map<IndependenceFact, Double> popConstraints_old = new HashMap<IndependenceFact, Double>();
////
////				for (IndependenceFact k : this.H_population.keySet()){					
////					if ((k.getX().equals(x) && k.getY().equals(y)) ||(k.getX().equals(y) && k.getY().equals(x))){
////						//						System.out.println("k: " + k );
////						//						System.out.println("p_old = " + this.H_population.get(k));
////
////						// convert set z to an array of indecies
////						Node[] _kz = k.getZ().toArray(new Node[k.getZ().size()]);
////						double pXYPopulation =  computeInd(bic_res, k.getX(), k.getY(), _kz);
////
////						//						System.out.println("p_new = " + pXYPopulation);
////
////						popConstraints_old.put(k, this.H_population.get(k));
////						popConstraints.put(k, pXYPopulation);
////
////						if (this.populationDsep.isIndependent(k.getX(), k.getY(), k.getZ())){
////							if (first){
////								pTotalPopulation = Math.log10(pXYPopulation);
////								first = false;
////							}
////							else{
////								//							System.out.println("INDEP -- P_before" + pTotalPopulation);
////								pTotalPopulation += Math.log10(pXYPopulation);
////								//							System.out.println("INDEP -- P_after" + pTotalPopulation);
////							}
////
////						}
////						else{
////							if (first){								
////								pTotalPopulation = Math.log10(1 - pXYPopulation);
////								first = false;
////							}
////							else{
////								//							System.out.println("DEP -- P_before: " + pTotalPopulation);
////								pTotalPopulation += Math.log10(1 - pXYPopulation);
////								//							System.out.println("DEP -- P_after: " + pTotalPopulation);
////							}
////						}
////					}
////				}
////				if(popConstraints.size()==0){
////					pTotalPopulation = Math.log10(0.5);
////				}
////
////				//				System.out.println("popConstraints_old: " + popConstraints_old);
////				//				System.out.println("popConstraints:     " + popConstraints);
////				//				System.out.println("pInd_is: " + pInd_is);
////				//				System.out.println("pRes: " + Math.pow(10,pTotalPopulation));
////				pInd = pTotalPopulation + Math.log10(pInd_is);
////				pInd = Math.pow(10, pInd);
////			}
////
////			H.put(key, pInd);
////
////		}else {
////			pInd = H.get(key);
////		}
////
////		//        System.out.println("pInd_old: " + pInd_old);
////		//        System.out.println("pInd: " + pInd);
////		//        System.out.println("--------------------");
////		double p = pInd; 
////
////		this.posterior = p;
////
////		boolean ind ;
////		if (this.threshold){
////			ind = (p >= cutoff);
////		}
////		else{
////			ind = RandomUtil.getInstance().nextDouble() < p;
////		}
////
////		if (ind) {
////			return true;
////		} else {
////			return false;
////		}
////	}
////
////	private double computeInd(BDeuScorWOprior bic, Node x, Node y, Node... z) {
////		double pInd = Double.NaN;
////		List<Node> _z = new ArrayList<>();
////		_z.add(x);
////		_z.add(y);
////		Collections.addAll(_z, z);
////
////		Graph indBN = new EdgeListGraph(_z);
////		for (Node n : z){
////			indBN.addDirectedEdge(n, x);
////			indBN.addDirectedEdge(n, y);
////		}
////
////		Graph depBN = new EdgeListGraph(_z);
////		depBN.addDirectedEdge(x, y);
////		for (Node n : z){
////			depBN.addDirectedEdge(n, x);
////			depBN.addDirectedEdge(n, y);
////		}
////		double indPrior = Math.log(0.5);
////		double indScore = scoreDag(indBN,bic);
////		//      double indScore = scoreDag(indBN, bic, false, null, null);
////		double scoreIndAll = indScore + indPrior;
////
////
////		double depScore = scoreDag(depBN, bic);
////		//      double depScore = scoreDag(depBN, bic, true, x, y);
////		double depPrior = Math.log(1 - indPrior);
////		double scoreDepAll = depScore + depPrior;
////
////		double scoreAll = lnXpluslnY(scoreIndAll, scoreDepAll);
////		//  	System.out.println("scoreDepAll: " + scoreDepAll);
////		//      System.out.println("scoreIndAll: " + scoreIndAll);
////		//      System.out.println("scoreAll: " + scoreAll);
////
////		pInd = Math.exp(scoreIndAll - scoreAll);
////
////		return pInd;
////	}
////
////
////	//        double indPrior = Math.log(0.5);
////	//        double indScore = scoreDag(indBN, bic_is);
////	//        double scoreIndAll = indScore + indPrior;
////	//
////	//        
////	//        double depScore = scoreDag(depBN, bic_is);
////	//        double depPrior = Math.log(1 - indPrior);
////	//        double scoreDepAll = depScore + depPrior;
////	//
////	//        double scoreAll = lnXpluslnY(scoreIndAll, scoreDepAll);
////	////    	System.out.println("scoreDepAll: " + scoreDepAll);
////	////        System.out.println("scoreIndAll: " + scoreIndAll);
////	////        System.out.println("scoreAll: " + scoreAll);
////	//
////	//        pInd = Math.exp(scoreIndAll - scoreAll);
////	//        
////	//        return pInd;
////	//	}
////
////	private void splitData(int[] parents){
////
////		int sampleSize = this.data.getNumRows();
////		int numVariables = this.data.getNumColumns();
////		ArrayList<Integer> rows_is = new ArrayList<>();
////		ArrayList<Integer> rows_res = new ArrayList<>();
////
////		for (int i = 0; i < sampleSize; i++){
////			int[] parentValuesTest = new int[parents.length];
////			int[] parentValuesCase = new int[parents.length];
////
////			for (int p = 0; p < parents.length ; p++){
////				parentValuesTest[p] =  test_array[0][parents[p]];
////				parentValuesCase[p] = data_array[i][parents[p]];
////			}
////			int [] row = new int[numVariables];
////			for (int j = 0; j < numVariables; j++){				
////				row[j] = data_array[i][j];
////			}
////			if (Arrays.equals(parentValuesCase, parentValuesTest)){
////				rows_is.add(i);
////			}
////			else{
////				rows_res.add(i);
////			}		
////		}
////
////		//		this.data_is = new BoxDataSet(((BoxDataSet) this.data).getDataBox(), this.data.getVariables()); 
////		this.data_is = new BoxDataSet((BoxDataSet)this.data);
////
////		//		this.data_res = new BoxDataSet(((BoxDataSet) this.data).getDataBox(), this.data.getVariables());
////		this.data_res = new BoxDataSet((BoxDataSet)this.data);
////
////		this.data_is.removeRows(rows_res.stream().mapToInt(i -> i).toArray());
////		this.data_res.removeRows(rows_is.stream().mapToInt(i -> i).toArray());
////		//		System.out.println("data     :" + this.data.getNumRows());
////		//		System.out.println("data is  :" + this.data_is.getNumRows());
////		//		System.out.println("data res :" + this.data_res.getNumRows());
////	}
////
////	public Map<IndependenceFact, Double> groupbyXY(Map<IndependenceFact, Double> H, Node x, Node y){
////		Map<IndependenceFact, Double> H_xy = new HashMap<IndependenceFact, Double>();
////		for (IndependenceFact k : H.keySet()){					
////			if ((k.getX().equals(x) && k.getY().equals(y) && k.getZ().size() > 0) ||(k.getX().equals(y) && k.getY().equals(x) && k.getZ().size() > 0)){
////				H_xy.put(k, H.get(k));
////			}
////		}
////		return H_xy;
////	}
////	
////	public void splitDatabyXY(DataSet data, DataSet data_xy, DataSet data_rest, Map<IndependenceFact, Double> H_xy){
////
////		int sampleSize = data.getNumRows();
////		int numVariables = data.getNumColumns();
////
////		Set<Integer> rows_is = new HashSet<>();
////		Set<Integer> rows_res = new HashSet<>();
////		for (int i = 0; i < data.getNumRows(); i++){
////			rows_res.add(i);
////		}
////		
////		Node x = null, y = null;
////		for(IndependenceFact f : H_xy.keySet()){
//////			System.out.println("f: " + f);
////			x = f.getX();
////			y = f.getY();
////			Node[] z = f.getZ().toArray(new Node[f.getZ().size()]);
////			int[] parents = new int[z.length];
////			for (int i = 0; i < z.length; i++) {
////				parents[i] = indices.get(z[i]);
////			}
////			
////			for (int i = 0; i < sampleSize; i++){
////				int[] parentValuesTest = new int[parents.length];
////				int[] parentValuesCase = new int[parents.length];
////
////				for (int p = 0; p < parents.length ; p++){
////					parentValuesTest[p] =  test_array[0][parents[p]];
////					parentValuesCase[p] = data_array[i][parents[p]];
////				}
////				
////				if (Arrays.equals(parentValuesCase, parentValuesTest)){
////					rows_is.add(i);
////					rows_res.remove(i);
////				}		
////			}
//////			System.out.println("rows_is: " + rows_is);
//////			System.out.println("rows_res: " + rows_res);
////		}
////
////		data_xy.removeRows(rows_res.stream().mapToInt(i -> i).toArray());
////		data_rest.removeRows(rows_is.stream().mapToInt(i -> i).toArray());
//////		System.out.println("data_xy: " + data_xy.getNumRows());
//////		System.out.println("data_rest: " + data_rest.getNumRows());
////
////	}
////
////	public double scoreDag(Graph dag, BDeuScorWOprior bic_is) {
////
////		double _score = 0.0;
////
////		for (Node y : dag.getNodes()) {
////			Set<Node> parents = new HashSet<>(dag.getParents(y));
////			int parentIndices[] = new int[parents.size()];
////			Iterator<Node> pi = parents.iterator();
////			int count = 0;
////
////			while (pi.hasNext()) {
////				Node nextParent = pi.next();
////				parentIndices[count++] = this.indices.get(nextParent);
////			}
////
////			int yIndex = this.indices.get(y);
////			_score += bic_is.localScore(yIndex, parentIndices);
////		}
////
////		return _score;
////	}
////
////	/**
////	 * Takes ln(x) and ln(y) as input, and returns ln(x + y)
////	 *
////	 * @param lnX is natural log of x
////	 * @param lnY is natural log of y
////	 * @return natural log of x plus y
////	 */
////	private static final int MININUM_EXPONENT = -1022;
////	protected double lnXpluslnY(double lnX, double lnY) {
////		double lnYminusLnX, temp;
////
////		if (lnY > lnX) {
////			temp = lnX;
////			lnX = lnY;
////			lnY = temp;
////		}
////
////		lnYminusLnX = lnY - lnX;
////
////		if (lnYminusLnX < MININUM_EXPONENT) {
////			return lnX;
////		} else {
////			return Math.log1p(Math.exp(lnYminusLnX)) + lnX;
////		}
////	}
////
////	//    public double probConstraint(BCInference.OP op, Node x, Node y, Node[] z) {
////	//
////	//        int _x = indices.get(x) + 1;
////	//        int _y = indices.get(y) + 1;
////	//
////	//        int[] _z = new int[z.length + 1];
////	//        _z[0] = z.length;
////	//        for (int i = 0; i < z.length; i++) {
////	//            _z[i + 1] = indices.get(z[i]) + 1;
////	//        }
////	//
////	//        return bci.probConstraint(op, _x, _y, _z);
////	//    }
////
////	@Override
////	public boolean isDependent(Node x, Node y, List<Node> z) {
////		Node[] _z = z.toArray(new Node[z.size()]);
////		return !isIndependent(x, y, _z);
////	}
////
////	@Override
////	public boolean isDependent(Node x, Node y, Node... z) {
////		return !isIndependent(x, y, z);
////	}
////
////	@Override
////	public double getPValue() {
////		return posterior;
////	}
////
////	@Override
////	public List<Node> getVariables() {
////		return nodes;
////	}
////
////	@Override
////	public Node getVariable(String name) {
////		for (Node node : nodes) {
////			if (name.equals(node.getName())) return node;
////		}
////
////		return null;
////	}
////
////	@Override
////	public List<String> getVariableNames() {
////		List<String> names = new ArrayList<>();
////
////		for (Node node : nodes) {
////			names.add(node.getName());
////		}
////		return names;
////	}
////
////	@Override
////	public boolean determines(List<Node> z, Node y) {
////		throw new UnsupportedOperationException();
////	}
////
////	@Override
////	public double getAlpha() {
////		throw new UnsupportedOperationException();
////	}
////
////	@Override
////	public void setAlpha(double alpha) {
////		throw new UnsupportedOperationException();
////	}
////
////	@Override
////	public DataModel getData() {
////		return data;
////	}
////
////	@Override
////	public ICovarianceMatrix getCov() {
////		return null;
////	}
////
////	@Override
////	public List<DataSet> getDataSets() {
////		return null;
////	}
////
////	@Override
////	public int getSampleSize() {
////		return 0;
////	}
////
////	@Override
////	public List<TetradMatrix> getCovMatrices() {
////		return null;
////	}
////
////	@Override
////	public double getScore() {
////		return getPValue();
////	}
////
////	public Map<IndependenceFact, Double> getH() {
////		return new HashMap<>(H);
////	}
////
////	//    private double probOp(BCInference.OP type, double pInd) {
////	//        double probOp;
////	//
////	//        if (BCInference.OP.independent == type) {
////	//            probOp = pInd;
////	//        } else {
////	//            probOp = 1.0 - pInd;
////	//        }
////	//
////	//        return probOp;
////	//    }
////
////	public double getPosterior() {
////		return posterior;
////	}
////
////	@Override
////	public boolean isVerbose() {
////		return verbose;
////	}
////
////	@Override
////	public void setVerbose(boolean verbose) {
////		this.verbose = verbose;
////	}
////
////	/**
////	 * @param noRandomizedGeneratingConstraints
////	 */
////	public void setThreshold(boolean noRandomizedGeneratingConstraints) {
////		this.threshold = noRandomizedGeneratingConstraints;
////	}
////
////	public void setCutoff(double cutoff) {
////		this.cutoff = cutoff;
////	}
////}
////
////
////
