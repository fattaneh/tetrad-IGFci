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
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.IndependenceFact;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TetradMatrix;
import edu.pitt.dbmi.algo.bayesian.constraint.inference.BCInference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Uses BCInference by Cooper and Bui to calculate probabilistic conditional independence judgments.
 *
 * @author Fattaneh June 11, 2019
 */
public class IndTestProbabilisticIS_OLD implements IndependenceTest {

	/**
	 * Calculates probabilities of independence for conditional independence facts.
	 */
	private BCInference bci = null;
	// Not
	private boolean threshold = false;

	/**
	 * The data set for which conditional  independence judgments are requested.
	 */
	private final DataSet data;
	private final int[][] data_array;
	private final int[][] cases;
	private int[][] data_is;
	private int[][] data_res;
	private final DataSet test;
	private final int[][] test_array;

	private final int[] nodeDimensions ;

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

	private Map<IndependenceFact, Double> H_population;
	private Graph populationGraph;
	private IndependenceTest populationDsep;
	private double posterior;
	private boolean verbose = false;

	private double cutoff = 0.5;

	//==========================CONSTRUCTORS=============================//
	/**
	 * Initializes the test using a discrete data sets.
	 */
	public IndTestProbabilisticIS_OLD(DataSet dataSet, DataSet test, Map<IndependenceFact, Double> H_population, Graph populationGraph) {
		if (!dataSet.isDiscrete()) {
			throw new IllegalArgumentException("Not a discrete data set.");

		}

		this.H_population = H_population;
		this.populationGraph = populationGraph;
		this.data = dataSet;
		this.test = test;

		// dsep test for population graph
		this.populationDsep = new IndTestDSep(this.populationGraph);
		
		//  convert the data and the test case to an array
		this.test_array = new int[test.getNumRows()][test.getNumColumns()];
		for (int i = 0; i < test.getNumRows(); i++) {
			for (int j = 0; j < test.getNumColumns(); j++) {
				test_array[i][j] = test.getInt(i, j);
			}
		}

		this.data_array = new int[dataSet.getNumRows()][dataSet.getNumColumns()];
		this.cases = new int[dataSet.getNumRows() + 1][dataSet.getNumColumns() + 2];

        
		for (int i = 0; i < dataSet.getNumRows(); i++) {
			for (int j = 0; j < dataSet.getNumColumns(); j++) {
				this.data_array[i][j] = dataSet.getInt(i, j);
				this.cases[i + 1][j + 1] = dataSet.getInt(i, j) + 1;
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
//		System.out.println("-----------------" );

		IndependenceFact key = new IndependenceFact(x, y, z);

		double pInd = Double.NaN;

		if (!H.containsKey(key)) {

//			System.out.println("ind key: " + key);

			// convert set z to an array of indecies
			int[] _z = new int[z.length];
			for (int i = 0; i < z.length; i++) {
				_z[i] = indices.get(z[i]);
			}

			if (_z.length ==0){
				BCInference bci = new BCInference(this.cases, this.nodeDimensions);
				pInd = probConstraint(bci, BCInference.OP.independent, x, y, z);
			}
			
			else{
				double pInd_is = Double.NaN;
				double pTotalPopulation = Double.NaN;
				boolean first = true;

				// split the data based on array _z
				splitData(_z);

				// compute BSC based on D that matches values of _z in the test case
				BCInference bci_is = new BCInference(this.data_is, this.nodeDimensions);
//				System.out.println("ind key: " + key);
				if(this.data_is.length>1){ 
					pInd_is = probConstraint(bci_is, BCInference.OP.independent, x, y, z);
				}
				else{
					pInd_is = 0.5;
				}

				// compute BSC based on D that does not match values of _z in the test case
				BCInference bci_res = new BCInference(this.data_res, this.nodeDimensions);
				Map<IndependenceFact, Double> popConstraints = new HashMap<IndependenceFact, Double>();
				Map<IndependenceFact, Double> popConstraints_old = new HashMap<IndependenceFact, Double>();

				for (IndependenceFact k : this.H_population.keySet()){					
					if ((k.getX().equals(x) && k.getY().equals(y)) ||(k.getX().equals(y) && k.getY().equals(x))){
//						System.out.println("k: " + k );
//						System.out.println("p_old = " + this.H_population.get(k));

						// convert set z to an array of indecies
						Node[] _kz = k.getZ().toArray(new Node[k.getZ().size()]);
						double pXYPopulation =  probConstraint(bci_res, BCInference.OP.independent, k.getX(), k.getY(), _kz);
						
//						System.out.println("p_new = " + pXYPopulation);

						popConstraints_old.put(k, this.H_population.get(k));
						popConstraints.put(k, pXYPopulation);
						
						if (this.populationDsep.isIndependent(k.getX(), k.getY(), k.getZ())){
							if (first){
								pTotalPopulation = Math.log10(pXYPopulation);
								first = false;
							}
							else{
//							System.out.println("INDEP -- P_before" + pTotalPopulation);
							pTotalPopulation += Math.log10(pXYPopulation);
//							System.out.println("INDEP -- P_after" + pTotalPopulation);
							}

						}
						else{
							if (first){								
								pTotalPopulation = Math.log10(1 - pXYPopulation);
								first = false;
							}
							else{
//							System.out.println("DEP -- P_before: " + pTotalPopulation);
							pTotalPopulation += Math.log10(1 - pXYPopulation);
//							System.out.println("DEP -- P_after: " + pTotalPopulation);
							}
						}
					}
				}
				if(popConstraints.size()==0){
					pTotalPopulation = Math.log10(0.5);
				}
				
//				System.out.println("popConstraints_old: " + popConstraints_old);
//				System.out.println("popConstraints:     " + popConstraints);
//				System.out.println("pInd_is: " + pInd_is);
//				System.out.println("pRes: " + Math.pow(10,pTotalPopulation));
				pInd = pTotalPopulation + Math.log10(pInd_is);
				pInd = Math.pow(10, pInd);
			}

			H.put(key, pInd);
		}

		else {
			pInd = H.get(key);
		}

		// re-score r_pop in H_pop that is also about x and y 
		double p = probOp(BCInference.OP.independent, pInd);

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

	//	private Map<IndependenceFact, Double> identifyPopConstraints_XY(Node x, Node y) {
	//		Map<IndependenceFact, Double> popConstraints = new HashMap<IndependenceFact, Double>();
	//		for (IndependenceFact k : this.H_population.keySet()){
	//			if ((k.getX().equals(x) && k.getY().equals(y)) ||(k.getX().equals(y) && k.getY().equals(x)))
	//				popConstraints.put(k, Double.NaN);
	//		}
	//		return popConstraints;
	//	}

	private void splitData(int[] parents){

		ArrayList<int[]> data_is = new ArrayList<int[]>(); 
		ArrayList<int[]> data_res = new ArrayList<int[]>(); 
		int sampleSize = this.data.getNumRows();
		int numVariables = this.data.getNumColumns();

		for (int i = 0; i < sampleSize; i++){
			int[] parentValuesTest = new int[parents.length];
			int[] parentValuesCase = new int[parents.length];

			for (int p = 0; p < parents.length ; p++){
				parentValuesTest[p] =  test_array[0][parents[p]];
				parentValuesCase[p] = data_array[i][parents[p]];
			}
			int [] row = new int[numVariables];
			for (int j = 0; j < numVariables; j++){				
				row[j] = data_array[i][j];
			}
			if (Arrays.equals(parentValuesCase, parentValuesTest)){
				data_is.add(row);
			}
			else{
				data_res.add(row);
			}		
		}

		this.data_is = new int[data_is.size() + 1][numVariables + 2];
		for (int i = 0; i < data_is.size(); i++){
			for (int j = 0; j < data_is.get(i).length; j++){
				this.data_is[i + 1][j + 1]= data_is.get(i)[j] + 1;
			}
		}

		this.data_res = new int [data_res.size() + 1][numVariables + 2];
		for (int i = 0; i < data_res.size(); i++){
			for (int j = 0; j < data_res.get(i).length; j++){
				this.data_res[i + 1][ j + 1]= data_res.get(i)[j] + 1;
			}
		}
		//		System.out.println(Arrays.deepToString(this.data_is));
		//		System.out.println(Arrays.deepToString(this.data_res));
	}


	public double probConstraint(BCInference bci, BCInference.OP op, Node x, Node y, Node[] z) {

		int _x = indices.get(x) + 1;
		int _y = indices.get(y) + 1;

		int[] _z = new int[z.length + 1];
		_z[0] = z.length;
		for (int i = 0; i < z.length; i++) {
			_z[i + 1] = indices.get(z[i]) + 1;
		}

		return bci.probConstraint(op, _x, _y, _z);
	}

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

	private double probOp(BCInference.OP type, double pInd) {
		double probOp;

		if (BCInference.OP.independent == type) {
			probOp = pInd;
		} else {
			probOp = 1.0 - pInd;
		}

		return probOp;
	}

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



