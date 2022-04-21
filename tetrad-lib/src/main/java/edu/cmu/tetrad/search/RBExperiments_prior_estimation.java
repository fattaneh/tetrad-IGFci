package edu.cmu.tetrad.search;

import static java.lang.Math.exp;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;
import java.util.regex.Pattern;

import static java.lang.Math.*;

import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusion;
import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusion;
import edu.cmu.tetrad.bayes.BayesIm;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.DirichletBayesIm;
import edu.cmu.tetrad.bayes.DirichletEstimator;
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TextTable;
import edu.pitt.dbmi.algo.bayesian.constraint.inference.BCInference;
import nu.xom.Builder;
import nu.xom.Document;
import nu.xom.ParsingException;

public class RBExperiments_prior_estimation {

	private int depth = 4;
	private static String directory;
	private static String algorithm;
	private PrintStream out;
	private static boolean completeRules = false;

	private List<Node> getLatents(Graph dag) {
		List<Node> latents = new ArrayList<>();
		for (Node n : dag.getNodes()) {
			if (n.getNodeType() == NodeType.LATENT) {
				latents.add(n);
			}
		}
		return latents;
	}

	public static void main(String[] args) throws IOException {
		// read and process input arguments
		Long seed = 1054147771L;
		String data_path =  "/Users/fattanehjabbari/CCD-Project/CS-BN/dissertation/BSC-Discrete/";
		// create an instance of class and run an experiment on it
		RBExperiments_prior_estimation.directory = "/Users/fattanehjabbari/CCD-Project/CS-BN/";
		RBExperiments_prior_estimation.algorithm = "FCI";
				
		double alpha = 0.05, cutoff = 0.5, edgesPerNode = 1.25, latent = 0.2,lower = 0.3, upper = 0.7;
		int numSim = 10, numModels = 100, numBootstrapSamples = 500;
		boolean threshold1 = false, threshold2 = true;

		int[] variableSize = new int[]{37};
		double[] edges = new double[]{1.25};
		double[] lv = new double[]{0.2};
		for (int var: variableSize){
			for (double epn: edges){
				for (double nlv: lv){
					RBExperiments_prior_estimation rbs = new RBExperiments_prior_estimation();
					rbs.experiment(numModels,alpha, threshold1, threshold2, cutoff, numBootstrapSamples, var, epn, nlv, numSim, data_path, seed);
				}
			}
		}

	}

	public void experiment(int numModels ,double alpha, boolean threshold1, boolean threshold2, double cutoff, int numBootstrapSamples,
			int numVars, double edgesPerNode, double latent, int numSim, String data_path, long seed){

		//		RandomUtil.getInstance().setSeed(seed + 10 * sim);
		//		RandomUtil.getInstance().setSeed(1454147771L);

		int minCat = 2;
		int maxCat = 4;
		final int numEdges = (int) (numVars * edgesPerNode);
		int numLatents = (int) Math.floor(numVars * latent);

		System.out.println("# nodes: " + numVars + ", # edges: "+ numEdges + ", # numLatents: "+ numLatents);
		
		double[][] prior = new double[numSim][numVars - 2];
		double[][] counts = new double[numSim][numVars - 2];

		// loop over simulations
		for (int s = 0; s < numSim; s++){
			RandomUtil.getInstance().setSeed(seed + 10 * s);
			BayesIm im = getBayesIM("Alarm");
			System.out.println("im:" + im);
			BayesPm pm = im.getBayesPm();
			Graph dag = pm.getDag();
			int LV = (int) Math.floor(numLatents * numVars);
			GraphUtils.fixLatents4(LV, dag);

//			System.out.println("simulation: " + s);
////			outlog.println("simulation: " + s);

//			List<Node> vars = createVariables(numVars);
//
//			// generate true BN and its parameters
//			Graph dag = GraphUtils.randomGraphRandomForwardEdges(vars, numLatents, numEdges, 15, 10, 10, false, true);
			System.out.println("Latent variables: " + getLatents(dag));
//			outlog.println("Latent variables: " + getLatents(dag));
//			BayesPm pm = new BayesPm(dag, minCat, maxCat);
//			MlBayesIm im = new MlBayesIm(pm, MlBayesIm.RANDOM);
			
			IndTestDSep dsep = new IndTestDSep(dag);
			Fci dsepFci = new Fci(dsep);
			dsepFci.setCompleteRuleSetUsed(RBExperiments_prior_estimation.completeRules);
			dsepFci.setDepth(this.depth);
			Graph PAG_True2 = dsepFci.search();
			System.out.println("FCI done");
			
//			System.out.println("PAG_True2.H: " + MapUtil.sortByValue(dsep.getH()));
//			Arrays.fill(prior, 0.0);

			Map<IndependenceFact, Double> H= dsep.getH();
			for(IndependenceFact f : H.keySet()){
				counts[s][f.getZ().size()] += 1.0;
				
				if (H.get(f) > 0.0){
					prior[s][f.getZ().size()] += 1.0;
				}
			}
			for(int p = 0; p < prior[s].length; p++){
				prior[s][p] /= counts[s][p];
			}
			System.out.println("priors: " + Arrays.toString(prior[s]));

		}
		System.out.println("counts: " + Arrays.deepToString(counts));
		System.out.println("priors: " + Arrays.deepToString(prior));
		double[] avg_prior = new double[numVars - 2];
		for(int i = 0; i < numVars - 2; i++){
			for(int j = 0; j < numSim; j++){
				avg_prior[i] += prior[j][i];
			}
			avg_prior[i] /= numSim;
		}
		System.out.println("avg priors: " + Arrays.toString(avg_prior));


	}

	private List<Node> createVariables(int numVars) {
		// create variables
		List<Node> vars = new ArrayList<>();
		for (int i = 0; i < numVars; i++) {
			vars.add(new DiscreteVariable("X" + i));
		}
		return vars;
	}
	
	private BayesIm getBayesIM(String type) {
		if ("Alarm".equals(type)) {
			return loadBayesIm("Alarm.xdsl", true);
		} else if ("Hailfinder".equals(type)) {
			return loadBayesIm("Hailfinder.xdsl", false);
		} else if ("Hepar".equals(type)) {
			return loadBayesIm("Hepar2.xdsl", true);
		} else if ("Win95".equals(type)) {
			return loadBayesIm("win95pts.xdsl", false);
		} else if ("Barley".equals(type)) {
			return loadBayesIm("barley.xdsl", false);
		}

		throw new IllegalArgumentException("Not a recogized Bayes IM type.");
	}
	private BayesIm loadBayesIm(String filename, boolean useDisplayNames) {
		try {
			Builder builder = new Builder();
			File dir = new File(this.directory + "/xdsl");
			File file = new File(dir, filename);
			Document document = builder.build(file);
			XdslXmlParser parser = new XdslXmlParser();
			parser.setUseDisplayNames(useDisplayNames);
			return parser.getBayesIm(document.getRootElement());
		} catch (ParsingException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
}



