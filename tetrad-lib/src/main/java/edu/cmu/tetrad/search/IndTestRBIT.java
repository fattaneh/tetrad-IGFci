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
import edu.cmu.tetrad.data.RandomSampler;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TetradMatrix;
import org.apache.commons.collections4.map.HashedMap;
import org.apache.commons.lang3.RandomUtils;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;

import static edu.cmu.tetrad.util.MathUtils.logChoose;
import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * Random Bayesian Independence Test
 *
 * @author Bryan Andrews
 */
public class IndTestRBIT implements IndependenceTest {

    private DataSet data;

    private double alpha;

    private double pValue;

    private double score;

    private int N;

    private TetradMatrix cov;

    private double L2P1 = log(2 * Math.PI) + 1;

    private boolean verbose = false;

	private double posterior;

	private boolean threshold = true;

	private double cutoff = 0.5;
	
    public IndTestRBIT(DataSet data) {
        this.data = data;
        this.N = data.getNumRows();
        this.cov = data.getCovarianceMatrix();
    }

    public IndependenceTest indTestSubset(List<Node> vars) {
        throw new UnsupportedOperationException();
    }

    public boolean isIndependent(Node x, Node y, List<Node> z) {
//        System.out.print(x + "_||_ " + y + "|" +  z);

        int[] n0 = new int[z.size()+2];
        int[] n1 = new int[z.size()];
        int[] d0 = new int[z.size()+1];
        int[] d1 = new int[z.size()+1];

        for (int i = 0; i < z.size(); i++) {
            n0[i] = data.getColumn(z.get(i));
            n1[i] = data.getColumn(z.get(i));
            d0[i] = data.getColumn(z.get(i));
            d1[i] = data.getColumn(z.get(i));
        }

        n0[z.size()] = data.getColumn(x);
        n0[z.size()+1] = data.getColumn(y);
        d0[z.size()] = data.getColumn(x);
        d1[z.size()] = data.getColumn(y);

        List<int[]> n = new ArrayList();
        List<int[]> d = new ArrayList();

        n.add(n0);
        n.add(n1);
        d.add(d0);
        d.add(d1);

        int llik = 0;

        for (int[] n_ : n) {
            double ldet = log(cov.getSelection(n_, n_).det());
            llik += -N/2.0 * (ldet + n_.length*L2P1) + n_.length/2.0;
        }

        for (int[] d_ : d) {
            double ldet = log(cov.getSelection(d_, d_).det());
            llik -= -N/2.0 * (ldet + d_.length*L2P1) + d_.length/2.0;
        }

        alpha = RandomUtils.nextDouble(0,1);
        pValue = Math.pow(exp(llik - log(N)/2.0) + 1, -1);
        score = alpha - pValue;

        double p = pValue; 
//        System.out.println(" = " + pValue);

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


    @Override
    public boolean isIndependent(Node x, Node y, Node... z) {
        System.out.print(x + "_||_ " + y + "|" +  z);
        List<Node> zList = Arrays.asList(z);
        return isIndependent(x, y, zList);
    }

    @Override
    public boolean isDependent(Node x, Node y, List<Node> z) {
        return !this.isIndependent(x, y, z);
    }

    @Override
    public boolean isDependent(Node x, Node y, Node... z) {
        List<Node> zList = Arrays.asList(z);
        return isDependent(x, y, zList);
    }

    @Override
    public List<Node> getVariables() {
        return data.getVariables();
    }

    @Override
    public List<String> getVariableNames() {
        List<Node> variables = getVariables();
        List<String> variableNames = new ArrayList<>();
        for (Node variable1 : variables) {
            variableNames.add(variable1.getName());
        }
        return variableNames;
    }

    @Override
    public Node getVariable(String name) {
        for (int i = 0; i < getVariables().size(); i++) {
            Node variable = getVariables().get(i);
            if (variable.getName().equals(name)) {
                return variable;
            }
        }

        return null;
    }

    @Override
    public boolean determines(List<Node> z, Node y) {
        return false;
    }

    @Override
    public double getPValue() {
        return posterior;
    }

    @Override
    public double getAlpha() {
        return alpha;
    }

    @Override
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public DataSet getData() {
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
        return N;
    }

    @Override
    public List<TetradMatrix> getCovMatrices() {
        return null;
    }

    @Override
    public double getScore() {
        return score;
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