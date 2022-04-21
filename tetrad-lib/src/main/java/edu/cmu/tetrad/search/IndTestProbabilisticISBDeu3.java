
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

import org.apache.commons.math3.special.Gamma;

/**
 * Uses BCInference by Cooper and Bui to calculate probabilistic conditional independence judgments.
 *
 * @author Fattaneh Jabbari 5/2020
 */
public class IndTestProbabilisticISBDeu3 implements IndependenceTest {

	private boolean threshold = false;

	/**
	 * The data set for which conditional  independence judgments are requested.
	 */
	private final DataSet data;
	private final DataSet test;
	private final int[][] data_array;
	private final int[][] test_array;
	//	private BDeuScorWOprior score;
	private double prior = 0.5;
//	private double d_IS_ind = 0.0, d_IS_dep = 0.0 , d_PW_ind = 0.0, d_PW_dep = 0.0;

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
	public IndTestProbabilisticISBDeu3(DataSet dataSet, DataSet test, Map<IndependenceFact, Double> H_population, Graph populationGraph) {
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


		this.nodeDimensions = new int[dataSet.getNumColumns()];

		for (int j = 0; j < dataSet.getNumColumns(); j++) {
			DiscreteVariable variable = (DiscreteVariable) (dataSet.getVariable(j));
			int numCategories = variable.getNumCategories();
			this.nodeDimensions[j] = numCategories;
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
//		System.out.println(x + " _||_ " + y +" | " + z);
		Node[] _z = z.toArray(new Node[z.size()]);
		return isIndependent(x, y, _z);
	}

	@Override
	public boolean isIndependent(Node x, Node y, Node... z) {
		IndependenceFact key = new IndependenceFact(x, y, z);
		 double pTotalPopulation = Double.NaN, pInd = Double.NaN, pInd_is = Double.NaN, pInd_pop = Double.NaN;

		if (!H.containsKey(key)) {
	
			// convert set z to an array of indicies
			int[] _z = new int[z.length];
			int[] values_z = new int[z.length];
			for (int i = 0; i < z.length; i++) {
				_z[i] = this.indices.get(z[i]);
				values_z[i] =  this.test_array[0][_z[i]];

			}

			if (_z.length == 0){
				pInd = computeInd(this.data, this.prior, false, x, y, z);
			}

			else{

				// split the data based on array _z
				DataSet data_is = new BoxDataSet((BoxDataSet) this.data);
				DataSet data_rest = new BoxDataSet((BoxDataSet) this.data);
				splitData(data_is, data_rest, _z, values_z);

				// compute BSC based on D that matches values of _z in the test case
//				double priorInd = 0.5;
				if(data_is.getNumRows() > 0){ 
//					if (data_rest.getNumRows() > 0){ 
					double priorInd = computeInd(data_rest, this.prior, true, x, y, z);
//					}
					pInd_is = computeInd_IS(data_is, priorInd, values_z, x, y, z);
//					double pInd_is2 = computeInd_IS(data_is, 0.5, values_z, x, y, z);
////					System.out.println(x + "_||_ " + y + "|" + _z);
//					System.out.println("prior: " + priorInd);					
//					System.out.println("pInd_is w/  p: " + pInd_is);
//					System.out.println("pInd_is wo/ p: " + pInd_is2);

				}	
				else{
					pInd_is = 0.5;
				}
				pInd = pInd_is;
			}


			H.put(key, pInd);

		}else {
			pInd = H.get(key);
		}

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


	public double computeInd(DataSet data, double priorI, boolean isPrior, Node x, Node y, Node... z) {
		BDeuScoreWOprior score = new BDeuScoreWOprior(data);

		List<Node> z_list = new ArrayList<>();
        Collections.addAll(z_list, z);
//		System.out.println("---------------------");
//		System.out.println(x + " _||_ " + y +" | " + z_list);

		double p_ind = Double.NaN, d_ind = 0.0, d_all = 0.0;

		int _x = this.indices.get(x);
		int _y = this.indices.get(y);
		int[] _z = new int[z.length];
//		System.out.println(_x + " _||_ " + _y +" | " + Arrays.toString(_z));

		
		int [] _xz = new int[_z.length + 1];
		int r = 1;
		ArrayList<CountObjects> d_z = new ArrayList<CountObjects>();
		for (int i = 0; i < z.length; i++) {
			_z[i] = this.indices.get(z[i]);
			_xz[i] = _z[i];
			r *= this.nodeDimensions[_z[i]];
		}
		_xz[_z.length] = _x;
//		Arrays.sort(_xz);
		int r2 = r * this.nodeDimensions[_x];
		

		double cellPrior_xz = score.getSamplePrior() / (this.nodeDimensions[_x] * r);
		double cellPrior_yz = score.getSamplePrior() / (this.nodeDimensions[_y] * r);
		double cellPrior_yxz = score.getSamplePrior() / (this.nodeDimensions[_y] * r2);
		double rowPrior_xz = score.getSamplePrior() / (r);
		double rowPrior_yz = score.getSamplePrior() / (r);
		double rowPrior_yxz = score.getSamplePrior() / (r2);

		
//		System.out.println("node dim: " + Arrays.toString(this.nodeDimensions));
//
//		System.out.println("_x_values: " + this.nodeDimensions[_x]);
//		System.out.println("_y_values: " + this.nodeDimensions[_y]);
		
//		double priorInd = Math.log(priorI) / r;  
//		double priorDep = Math.log(1.0 - Math.exp(priorInd));
		
		double priorInd = Math.log(priorI) / r; ;// Math.log(Math.pow(priorI, z.length+1)) / r;
		double priorDep = Math.log(1.0 - Math.exp(priorInd));

//		double d_xz = 0.0, d_yz = 0.0, d_yxz = 0.0;
		for (int j = 0; j < r; j++) {
			double d_xz = 0.0, d_yz = 0.0, d_yxz = 0.0;
			int[] _z_dim = getDimension(_z);
			int[] _z_values = getParentValues(_y, j, _z_dim);
//			System.out.println("_z_values: " + Arrays.toString(_z_values));
			DataSet data_is = new BoxDataSet((BoxDataSet) this.data);
			splitData(data_is, _z, _z_values);
//			System.out.println("data_is: " + data_is.getNumRows());

			BDeuScoreWOprior score_z = new BDeuScoreWOprior(data_is);
			CountObjects counts_xz = score_z.localCounts(_x, _z);
			CountObjects counts_yz = score_z.localCounts(_y, _z);
			CountObjects counts_yxz = score_z.localCounts(_y, _xz);
//			System.out.println("d_xz: " + Arrays.deepToString(counts_xz.n_jk));
//			System.out.println("d_yz: " + Arrays.deepToString(counts_yz.n_jk));
//			System.out.println("d_yxz: " + Arrays.deepToString(counts_yxz.n_jk));
//			System.out.println("d_yxz: " + Arrays.toString(counts_yxz.n_j));
			// compute d_{x|z}
			int [] n_j = counts_xz.n_j;
			int [][] n_jk = counts_xz.n_jk;
//			System.out.println("d_xz b: " + d_xz);
			d_xz -= Gamma.logGamma(rowPrior_xz + n_j[j]);
			d_xz += Gamma.logGamma(rowPrior_xz);

			for (int k = 0; k < this.nodeDimensions[_x]; k++) {
//				System.out.println("n_jk x: " + Arrays.deepToString(n_jk));
//				System.out.println("j: " + j + ", k: " + k);
//				System.out.println("x: " + x + ", index: " + _x);
//				System.out.println("this.nodeDimensions[_x]: " + this.nodeDimensions[_x]);

				d_xz += Gamma.logGamma(cellPrior_xz + n_jk[j][k]);
				d_xz -= Gamma.logGamma(cellPrior_xz);
			}
//			System.out.println("d_xz a: " + d_xz);

			// compute d_{y|z}
			n_j = counts_yz.n_j;
			n_jk = counts_yz.n_jk;
//			System.out.println("d_yz b: " + d_yz);
			d_yz-= Gamma.logGamma(rowPrior_yz + n_j[j]);
			d_yz += Gamma.logGamma(rowPrior_yz);
			
			for (int k = 0; k < this.nodeDimensions[_y]; k++) {
//				System.out.println("n_jk y: " + Arrays.deepToString(n_jk));
//				System.out.println("j: " + j + ", k: " + k);
//				System.out.println("y: " + y + ", index: " + _y);
//				System.out.println("this.nodeDimensions[_y]: " + this.nodeDimensions[_y]);

				d_yz += Gamma.logGamma(cellPrior_yz + n_jk[j][k]);
				d_yz -= Gamma.logGamma(cellPrior_yz);
			}
//			d_yz += priorInd;

//			System.out.println("d_yz a: " + d_yz);

			// compute d_{y|x,z}
			n_j = counts_yxz.n_j;
			n_jk = counts_yxz.n_jk;

			int[] _xz_values = new int[_z_values.length + 1];
			int[] _xz_dim = new int[_z_dim.length + 1];
			for (int v = 0; v < _z.length; v++){
				_xz_values[v] = _z_values[v];
				_xz_dim[v] = _z_dim[v];
			}
//			System.out.println("_xz_values b: " + Arrays.toString(_xz_values));
//
			for (int j2 = 0; j2 < this.nodeDimensions[_x]; j2++) {
				_xz_values[_z.length] = j2;
//				System.out.println("_xz_values a: " + Arrays.toString(_xz_values));
			
				_xz_dim[_z_values.length] = this.nodeDimensions[_x];
				int rowIndex = getRowIndex(_xz_dim, _xz_values);
				d_yxz-= Gamma.logGamma(rowPrior_yxz + n_j[rowIndex]);
				d_yxz += Gamma.logGamma(rowPrior_yxz);
				for (int k = 0; k < this.nodeDimensions[_y]; k++) {
					d_yxz += Gamma.logGamma(cellPrior_yxz + n_jk[rowIndex][k]);
					d_yxz -= Gamma.logGamma(cellPrior_yxz);
				}
				//				d_yxz += priorDep;
			}
			d_ind += priorInd + (d_xz + d_yz);
			d_all += lnXpluslnY (priorInd + (d_xz + d_yz), priorDep + (d_xz + d_yxz));
		}

		
		if (isPrior){
			p_ind = Math.exp((d_ind - d_all)/(r-1));
		}
		else{
			p_ind = Math.exp(d_ind - d_all);
		}
//		System.out.println("p_ind: " + p_ind);
		return p_ind;
	}
//	
////		BDeuScorWOprior score = new BDeuScorWOprior(data);
////
////		List<Node> z_list = new ArrayList<>();
////		Collections.addAll(z_list, z);
////		//		System.out.println("---------------------");
////		//		System.out.println(x + " _||_ " + y +" | " + z_list);
////
////		double p_ind = Double.NaN, d_ind = 0.0, d_all = 0.0;
////
////		int _x = this.indices.get(x);
////		int _y = this.indices.get(y);
////		int[] _z = new int[z.length];
////		int [] _xz = new int[_z.length + 1];
////		int r = 1;
//////		ArrayList<CountObjects> d_z = new ArrayList<CountObjects>();
////		for (int i = 0; i < z.length; i++) {
////			_z[i] = this.indices.get(z[i]);
////			_xz[i] = _z[i];
////			r *= this.nodeDimensions[_z[i]];
////		}
////		_xz[_z.length] = _x;
////		//		Arrays.sort(_xz);
////		int r2 = r * this.nodeDimensions[_x];
////
////
////		double cellPrior_xz = score.getSamplePrior() / (this.nodeDimensions[_x] * r);
////		double cellPrior_yz = score.getSamplePrior() / (this.nodeDimensions[_y] * r);
////		double cellPrior_yxz = score.getSamplePrior() / (this.nodeDimensions[_y] * r2);
////		double rowPrior_xz = score.getSamplePrior() / (r);
////		double rowPrior_yz = score.getSamplePrior() / (r);
////		double rowPrior_yxz = score.getSamplePrior() / (r2);
////
////
////		//		System.out.println("_x_values: " + this.nodeDimensions[_x]);
////		//		System.out.println("_y_values: " + this.nodeDimensions[_y]);
////		double priorInd = Math.log(this.prior) / r;  
////		double priorDep = Math.log(1 - Math.exp(priorInd));
////		//		double d_xz = 0.0, d_yz = 0.0, d_yxz = 0.0;
////		for (int j = 0; j < r; j++) {
////			double d_xz = 0.0, d_yz = 0.0, d_yxz = 0.0;
////			int[] _z_dim = getDimension(_z);
////			int[] _z_values = getParentValues(_y, j, _z_dim);
////			//			System.out.println("_z_values: " + Arrays.toString(_z_values));
////			DataSet data_is = new BoxDataSet((BoxDataSet) data);
////			splitData(data_is, _z, _z_values);
////			//			System.out.println("data_is: " + data_is.getNumRows());
////
////			BDeuScorWOprior score_z = new BDeuScorWOprior(data_is);
////			CountObjects counts_xz = score_z.localCounts(_x, _z);
////			CountObjects counts_yz = score_z.localCounts(_y, _z);
////			CountObjects counts_yxz = score_z.localCounts(_y, _xz);
////			//			System.out.println("d_xz: " + Arrays.deepToString(counts_xz.n_jk));
////			//			System.out.println("d_yz: " + Arrays.deepToString(counts_yz.n_jk));
////			//			System.out.println("d_yxz: " + Arrays.deepToString(counts_yxz.n_jk));
////			//			System.out.println("d_yxz: " + Arrays.toString(counts_yxz.n_j));
////			// compute d_{x|z}
////			int [] n_j = counts_xz.n_j;
////			int [][] n_jk = counts_xz.n_jk;
////			//			System.out.println("d_xz b: " + d_xz);
////			d_xz -= Gamma.logGamma(rowPrior_xz + n_j[j]);
////			d_xz += Gamma.logGamma(rowPrior_xz);
////			for (int k = 0; k < this.nodeDimensions[_x]; k++) {
////				d_xz += Gamma.logGamma(cellPrior_xz + n_jk[j][k]);
////				d_xz -= Gamma.logGamma(cellPrior_xz);
////			}
////			//			System.out.println("d_xz a: " + d_xz);
////
////			// compute d_{y|z}
////			n_j = counts_yz.n_j;
////			n_jk = counts_yz.n_jk;
////			//			System.out.println("d_yz b: " + d_yz);
////			d_yz-= Gamma.logGamma(rowPrior_yz + n_j[j]);
////			d_yz += Gamma.logGamma(rowPrior_yz);
////			for (int k = 0; k < this.nodeDimensions[_y]; k++) {
////				d_yz += Gamma.logGamma(cellPrior_yz + n_jk[j][k]);
////				d_yz -= Gamma.logGamma(cellPrior_yz);
////			}
////			//			d_yz += priorInd;
////
////			//			System.out.println("d_yz a: " + d_yz);
////
////			// compute d_{y|x,z}
////			n_j = counts_yxz.n_j;
////			n_jk = counts_yxz.n_jk;
////
////			int[] _xz_values = new int[_z_values.length + 1];
////			int[] _xz_dim = new int[_z_dim.length + 1];
////			for (int v = 0; v < _z.length; v++){
////				_xz_values[v] = _z_values[v];
////				_xz_dim[v] = _z_dim[v];
////			}
////			//			System.out.println("_xz_values b: " + Arrays.toString(_xz_values));
////			//
////			for (int j2 = 0; j2 < this.nodeDimensions[_x]; j2++) {
////				_xz_values[_z.length] = j2;
////				//				System.out.println("_xz_values a: " + Arrays.toString(_xz_values));
////			}
////			_xz_dim[_z_values.length] = this.nodeDimensions[_x];
////			int rowIndex = getRowIndex(_xz_dim, _xz_values);
////			d_yxz-= Gamma.logGamma(rowPrior_yxz + n_j[rowIndex]);
////			d_yxz += Gamma.logGamma(rowPrior_yxz);
////			for (int k = 0; k < this.nodeDimensions[_y]; k++) {
////				d_yxz += Gamma.logGamma(cellPrior_yxz + n_jk[rowIndex][k]);
////				d_yxz -= Gamma.logGamma(cellPrior_yxz);
////			}
////			//				d_yxz += priorDep;
////
////			d_ind += priorInd + (d_xz + d_yz);
////			d_all += lnXpluslnY (priorInd + (d_xz + d_yz), priorDep + (d_xz + d_yxz));
////		}
////
////		p_ind = Math.exp(d_ind - d_all);
////		//		System.out.println("p_ind: " + p_ind);
////		return p_ind;
//	}

	public double computeInd_IS(DataSet data, double priorI, int[] _z_values, Node x, Node y, Node... z) {

		BDeuScoreWOprior score = new BDeuScoreWOprior(data);

		List<Node> z_list = new ArrayList<>();
		Collections.addAll(z_list, z);
		double d_ind = 0.0, d_all = 0.0;

		int _x = this.indices.get(x);
		int _y = this.indices.get(y);
		int[] _z = new int[z.length];
		int [] _xz = new int[_z.length + 1];
		int r = 1;
//		ArrayList<CountObjects> d_z = new ArrayList<CountObjects>();
		for (int i = 0; i < z.length; i++) {
			_z[i] = this.indices.get(z[i]);
			_xz[i] = _z[i];
//			r *= this.nodeDimensions[_z[i]];
		}
		_xz[_z.length] = _x;
		//	Arrays.sort(_xz);
		int r2 = r * this.nodeDimensions[_x];


		double cellPrior_xz = score.getSamplePrior() / (this.nodeDimensions[_x] * r);
		double cellPrior_yz = score.getSamplePrior() / (this.nodeDimensions[_y] * r);
		double cellPrior_yxz = score.getSamplePrior() / (this.nodeDimensions[_y] * r2);
		double rowPrior_xz = score.getSamplePrior() / (r);
		double rowPrior_yz = score.getSamplePrior() / (r);
		double rowPrior_yxz = score.getSamplePrior() / (r2);


		//	System.out.println("_x_values: " + this.nodeDimensions[_x]);
		//	System.out.println("_y_values: " + this.nodeDimensions[_y]);
		double priorInd = Math.log(priorI) / r;  
		double priorDep = Math.log(1.0 - priorInd);
		//	double d_xz = 0.0, d_yz = 0.0, d_yxz = 0.0;
		for (int j = 0; j < r; j++) {
			double d_xz = 0.0, d_yz = 0.0, d_yxz = 0.0;
			int[] _z_dim = getDimension(_z);
			//		int[] _z_values = getParentValues(_y, j, _z_dim);
			//		System.out.println("_z_values: " + Arrays.toString(_z_values));
			DataSet data_is = new BoxDataSet((BoxDataSet) data);
//			splitData(data_is, _z, _z_values);
			//		System.out.println("data_is: " + data_is.getNumRows());

			BDeuScoreWOprior score_z = new BDeuScoreWOprior(data_is);
			CountObjects counts_xz = score_z.localCounts(_x, _z);
			CountObjects counts_yz = score_z.localCounts(_y, _z);
			CountObjects counts_yxz = score_z.localCounts(_y, _xz);
			//		System.out.println("d_xz: " + Arrays.deepToString(counts_xz.n_jk));
			//		System.out.println("d_yz: " + Arrays.deepToString(counts_yz.n_jk));
			//		System.out.println("d_yxz: " + Arrays.deepToString(counts_yxz.n_jk));
			//		System.out.println("d_yxz: " + Arrays.toString(counts_yxz.n_j));
			// compute d_{x|z}
			int [] n_j = counts_xz.n_j;
			int [][] n_jk = counts_xz.n_jk;
			//		System.out.println("d_xz b: " + d_xz);
			d_xz -= Gamma.logGamma(rowPrior_xz + n_j[j]);
			d_xz += Gamma.logGamma(rowPrior_xz);
			for (int k = 0; k < this.nodeDimensions[_x]; k++) {
				d_xz += Gamma.logGamma(cellPrior_xz + n_jk[j][k]);
				d_xz -= Gamma.logGamma(cellPrior_xz);
			}
			//		System.out.println("d_xz a: " + d_xz);

			// compute d_{y|z}
			n_j = counts_yz.n_j;
			n_jk = counts_yz.n_jk;
			//		System.out.println("d_yz b: " + d_yz);
			d_yz-= Gamma.logGamma(rowPrior_yz + n_j[j]);
			d_yz += Gamma.logGamma(rowPrior_yz);
			for (int k = 0; k < this.nodeDimensions[_y]; k++) {
				d_yz += Gamma.logGamma(cellPrior_yz + n_jk[j][k]);
				d_yz -= Gamma.logGamma(cellPrior_yz);
			}
			//		d_yz += priorInd;

			//		System.out.println("d_yz a: " + d_yz);

			// compute d_{y|x,z}
			n_j = counts_yxz.n_j;
			n_jk = counts_yxz.n_jk;

			int[] _xz_values = new int[_z_values.length + 1];
			int[] _xz_dim = new int[_z_dim.length + 1];
			for (int v = 0; v < _z.length; v++){
				_xz_values[v] = _z_values[v];
				_xz_dim[v] = _z_dim[v];
			}
			//		System.out.println("_xz_values b: " + Arrays.toString(_xz_values));
			//
			for (int j2 = 0; j2 < this.nodeDimensions[_x]; j2++) {
				_xz_values[_z.length] = j2;
				//			System.out.println("_xz_values a: " + Arrays.toString(_xz_values));
			}
			_xz_dim[_z_values.length] = this.nodeDimensions[_x];
			int rowIndex = getRowIndex(_xz_dim, _xz_values);
			d_yxz-= Gamma.logGamma(rowPrior_yxz + n_j[rowIndex]);
			d_yxz += Gamma.logGamma(rowPrior_yxz);
			for (int k = 0; k < this.nodeDimensions[_y]; k++) {
				d_yxz += Gamma.logGamma(cellPrior_yxz + n_jk[rowIndex][k]);
				d_yxz -= Gamma.logGamma(cellPrior_yxz);
			}
			//			d_yxz += priorDep;

			d_ind += priorInd + (d_xz + d_yz);
			d_all += lnXpluslnY (priorInd + (d_xz + d_yz), priorDep + (d_xz + d_yxz));
		}

		double p_ind = Math.exp(d_ind - d_all);
		//	System.out.println("p_ind: " + p_ind);
		return p_ind;
	}
//	public double computeInd_pop(DataSet data, double prior, Node x, Node y, Node... z) {
//		BDeuScorWOprior score = new BDeuScorWOprior(data);
//
//		List<Node> z_list = new ArrayList<>();
//		Collections.addAll(z_list, z);
//		double d_ind = 0.0, d_all = 0.0;
//		System.out.println(x + " _||_ " + y +" | " + z_list);
//		int _x = this.indices.get(x);
//		int _y = this.indices.get(y);
//		int[] _z = new int[z.length];
//		int [] _xz = new int[_z.length + 1];
//		int r = 1;
////		ArrayList<CountObjects> d_z = new ArrayList<CountObjects>();
//		for (int i = 0; i < z.length; i++) {
//			_z[i] = this.indices.get(z[i]);
//			_xz[i] = _z[i];
//			r *= this.nodeDimensions[_z[i]];
//		}
//		_xz[_z.length] = _x;
//		//		Arrays.sort(_xz);
//		int r2 = r * this.nodeDimensions[_x];
//
//
//		double cellPrior_xz = score.getSamplePrior() / (this.nodeDimensions[_x] * r);
//		double cellPrior_yz = score.getSamplePrior() / (this.nodeDimensions[_y] * r);
//		double cellPrior_yxz = score.getSamplePrior() / (this.nodeDimensions[_y] * r2);
//		double rowPrior_xz = score.getSamplePrior() / (r);
//		double rowPrior_yz = score.getSamplePrior() / (r);
//		double rowPrior_yxz = score.getSamplePrior() / (r2);
//
//
//		System.out.println("_x_values: " + this.nodeDimensions[_x]);
//		System.out.println("_y_values: " + this.nodeDimensions[_y]);		
//		System.out.println("_z_values: " + r);
//
//		double priorInd = Math.log(prior) / r;  
//		double priorDep = Math.log(1 - Math.exp(priorInd));
//		//		double d_xz = 0.0, d_yz = 0.0, d_yxz = 0.0;
//		for (int j = 0; j < r; j++) {
//			System.out.println("j: " + j);
//			double d_xz = 0.0, d_yz = 0.0, d_yxz = 0.0;
//			int[] _z_dim = getDimension(_z);
//			int[] _z_values = getParentValues(_y, j, _z_dim);
////						System.out.println("_z_values: " + Arrays.toString(_z_values));
//			DataSet data_is = new BoxDataSet((BoxDataSet) data);
//			splitData(data_is, _z, _z_values);
//			//			System.out.println("data_is: " + data_is.getNumRows());
//
//			BDeuScorWOprior score_z = new BDeuScorWOprior(data_is);
//			CountObjects counts_xz = score_z.localCounts(_x, _z);
//			CountObjects counts_yz = score_z.localCounts(_y, _z);
//			CountObjects counts_yxz = score_z.localCounts(_y, _xz);
//			System.out.println("d_xz: " + Arrays.deepToString(counts_xz.n_jk));
//			System.out.println("d_yz: " + Arrays.deepToString(counts_yz.n_jk));
//			System.out.println("d_yxz: " + Arrays.deepToString(counts_yxz.n_jk));
//			System.out.println("d_yxz: " + Arrays.toString(counts_yxz.n_j));
//			// compute d_{x|z}
//			int [] n_j = counts_xz.n_j;
//			int [][] n_jk = counts_xz.n_jk;
//			//			System.out.println("d_xz b: " + d_xz);
//			d_xz -= Gamma.logGamma(rowPrior_xz + n_j[j]);
//			d_xz += Gamma.logGamma(rowPrior_xz);
//			for (int k = 0; k < this.nodeDimensions[_x]; k++) {
//				d_xz += Gamma.logGamma(cellPrior_xz + n_jk[j][k]);
//				d_xz -= Gamma.logGamma(cellPrior_xz);
//			}
//			//			System.out.println("d_xz a: " + d_xz);
//
//			// compute d_{y|z}
//			n_j = counts_yz.n_j;
//			n_jk = counts_yz.n_jk;
//			//			System.out.println("d_yz b: " + d_yz);
//			d_yz-= Gamma.logGamma(rowPrior_yz + n_j[j]);
//			d_yz += Gamma.logGamma(rowPrior_yz);
//			for (int k = 0; k < this.nodeDimensions[_y]; k++) {
//				d_yz += Gamma.logGamma(cellPrior_yz + n_jk[j][k]);
//				d_yz -= Gamma.logGamma(cellPrior_yz);
//			}
//			//			d_yz += priorInd;
//
//			//			System.out.println("d_yz a: " + d_yz);
//
//			// compute d_{y|x,z}
//			n_j = counts_yxz.n_j;
//			n_jk = counts_yxz.n_jk;
//
//			int[] _xz_values = new int[_z_values.length + 1];
//			int[] _xz_dim = new int[_z_dim.length + 1];
//			for (int v = 0; v < _z.length; v++){
//				_xz_values[v] = _z_values[v];
//				_xz_dim[v] = _z_dim[v];
//			}
//			//			System.out.println("_xz_values b: " + Arrays.toString(_xz_values));
//			//
//			for (int j2 = 0; j2 < this.nodeDimensions[_x]; j2++) {
//				_xz_values[_z.length] = j2;
//				//				System.out.println("_xz_values a: " + Arrays.toString(_xz_values));
//			}
//			_xz_dim[_z_values.length] = this.nodeDimensions[_x];
//			int rowIndex = getRowIndex(_xz_dim, _xz_values);
//			d_yxz-= Gamma.logGamma(rowPrior_yxz + n_j[rowIndex]);
//			d_yxz += Gamma.logGamma(rowPrior_yxz);
//			for (int k = 0; k < this.nodeDimensions[_y]; k++) {
//				d_yxz += Gamma.logGamma(cellPrior_yxz + n_jk[rowIndex][k]);
//				d_yxz -= Gamma.logGamma(cellPrior_yxz);
//			}
//			//				d_yxz += priorDep;
//
//			d_ind += priorInd + (d_xz + d_yz);
//			d_all += lnXpluslnY (priorInd + (d_xz + d_yz), priorDep + (d_xz + d_yxz));
//		}
//
//		double p_ind = Math.exp(d_ind - d_all);
//		System.out.println("p_ind: " + p_ind);
//		System.out.println("---------------");
//		return p_ind;
//
//	}

	private int[] getDimension(int[] parents) {
		int[] dims = new int[parents.length];

		for (int p = 0; p < parents.length; p++) {
			dims[p] = this.nodeDimensions[parents[p]];
		}
		return dims;
	}

	private void splitData(DataSet data_xy, int[] parents, int[] parentValues){
		int sampleSize = data.getNumRows();

		Set<Integer> rows_is = new HashSet<>();
		Set<Integer> rows_res = new HashSet<>();

		for (int i = 0; i < data.getNumRows(); i++){
			rows_res.add(i);
		}

		for (int i = 0; i < sampleSize; i++){
//			int[] parentValuesTest = new int[parents.length];
			int[] parentValuesCase = new int[parents.length];

			for (int p = 0; p < parents.length ; p++){
				parentValuesCase[p] = data_array[i][parents[p]];
			}

			if (Arrays.equals(parentValuesCase, parentValues)){
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
		//		System.out.println("data_xy: " + data_xy.getNumRows());
		//		System.out.println("data_rest: " + data_rest.getNumRows());

	}
	public int[] getParentValues(int nodeIndex, int rowIndex, int[] dims) {
		int[] values = new int[dims.length];

		for (int i = dims.length - 1; i >= 0; i--) {
			values[i] = rowIndex % dims[i];
			rowIndex /= dims[i];
		}

		return values;
	}
	public int getRowIndex(int[] dim, int[] values) {
		int rowIndex = 0;
		for (int i = 0; i < dim.length; i++) {
			rowIndex *= dim[i];
			rowIndex += values[i];
		}
		return rowIndex;
	}

	
	private void splitData(DataSet data_xy, DataSet data_rest, int[] parents, int[] parentValuesTest){
		int sampleSize = data.getNumRows();

		Set<Integer> rows_is = new HashSet<>();
		Set<Integer> rows_res = new HashSet<>();

		for (int i = 0; i < data.getNumRows(); i++){
			rows_res.add(i);
		}

		for (int i = 0; i < sampleSize; i++){
//			int[] parentValuesTest = new int[parents.length];
			int[] parentValuesCase = new int[parents.length];

			for (int p = 0; p < parents.length ; p++){
//				parentValuesTest[p] =  this.test_array[0][parents[p]];
				parentValuesCase[p] = this.data_array[i][parents[p]];
			}

			if (Arrays.equals(parentValuesCase, parentValuesTest)){
				rows_is.add(i);
				rows_res.remove(i);
			}		
		}

		int[] is_array = new int[rows_is.size()];
		int c = 0;
		for(int row : rows_is) is_array[c++] = row;

		int[] rest_array = new int[rows_res.size()];
		c = 0;
		for(int row : rows_res) rest_array[c++] = row;

		Arrays.sort(is_array);
		Arrays.sort(rest_array);

		data_xy.removeRows(rest_array);
		data_rest.removeRows(is_array);
//		System.out.println("data_xy: " + data_xy.getNumRows());
//		System.out.println("data_rest: " + data_rest.getNumRows());

	}


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
				parents[i] = this.indices.get(z[i]);
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

