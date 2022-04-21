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
public class IndTestProbabilisticISBic implements IndependenceTest {

	private boolean threshold = false;

    /**
     * The data set for which conditional  independence judgments are requested.
     */
    private final DataSet data;
	private final DataSet test;
	private DataSet data_is;
	private DataSet data_res;
	private final int[][] data_array;
	private final int[][] test_array;
	
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
	public IndTestProbabilisticISBic(DataSet dataSet, DataSet test, Map<IndependenceFact, Double> H_population, Graph populationGraph) {
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
				BicScore bic = new BicScore(this.data);
				pInd = computeInd(bic, x, y, z);
			}
			
			else{
				double pInd_is = Double.NaN;
				double pTotalPopulation = Double.NaN;
				boolean first = true;

				// split the data based on array _z
				splitData(_z);
				
				// compute BSC based on D that matches values of _z in the test case
//				System.out.println("ind key: " + key);
				if(this.data_is.getNumRows() > 0){ 
					BicScore bic_is = new BicScore(this.data_is);
					pInd_is = computeInd(bic_is, x, y, z);
				}
				else{
					pInd_is = 0.5;
				}

				// compute BSC based on D that does not match values of _z in the test case
				BicScore bic_res = new BicScore(this.data_res);
				Map<IndependenceFact, Double> popConstraints = new HashMap<IndependenceFact, Double>();
				Map<IndependenceFact, Double> popConstraints_old = new HashMap<IndependenceFact, Double>();

				for (IndependenceFact k : this.H_population.keySet()){					
					if ((k.getX().equals(x) && k.getY().equals(y)) ||(k.getX().equals(y) && k.getY().equals(x))){
//						System.out.println("k: " + k );
//						System.out.println("p_old = " + this.H_population.get(k));

						// convert set z to an array of indecies
						Node[] _kz = k.getZ().toArray(new Node[k.getZ().size()]);
						double pXYPopulation =  computeInd(bic_res, k.getX(), k.getY(), _kz);
						
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

	private double computeInd(BicScore bic, Node x, Node y, Node... z) {
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
//      double indScore = scoreDag(indBN);
      double indScore = scoreDag(indBN, bic, false, null, null);
      double scoreIndAll = indScore + indPrior;

      
//      double depScore = scoreDag(depBN);
      double depScore = scoreDag(depBN, bic, true, x, y);
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
	
	public double scoreDag(Graph dag, BicScore bic, boolean isDep, Node xx, Node yy) {

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
//            if (isDep && yIndex == this.indices.get(yy)){ 
//            	_score += bic.localScore(yIndex, parentIndices, 1.5/(this.nodeDimensions[this.indices.get(xx)]));
//            	
//            }
//            else{
            _score += bic.localScore(yIndex, parentIndices); 	
//            }
           
       
        }

        return _score;
    }
	private void splitData(int[] parents){
//		System.out.println("splitData");
		
		int sampleSize = this.data.getNumRows();
		int numVariables = this.data.getNumColumns();
		ArrayList<Integer> rows_is = new ArrayList<>();
		ArrayList<Integer> rows_res = new ArrayList<>();
		
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
				rows_is.add(i);
			}
			else{
				rows_res.add(i);
			}		
		}
		
		this.data_is = new BoxDataSet(((BoxDataSet) this.data).getDataBox(), this.data.getVariables()); 
		this.data_res = new BoxDataSet(((BoxDataSet) this.data).getDataBox(), this.data.getVariables());

		this.data_is.removeRows(rows_res.stream().mapToInt(i -> i).toArray());
		this.data_res.removeRows(rows_is.stream().mapToInt(i -> i).toArray());
//		System.out.println("data     :" + this.data.getNumRows());
//		System.out.println("data is  :" + this.data_is.getNumRows());
//		System.out.println("data res :" + this.data_res.getNumRows());
	}
	
	public double scoreDag(Graph dag, BicScore bic_is) {

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



