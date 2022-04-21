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

public class RBExperiments_sim_old {

	private static int depth;
	private static String directory;
	private static String algorithm;
	private PrintStream out;
	private static boolean completeRules;
	private static double prior;

	private static class MapUtil {
		public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
			List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>(map.entrySet());
			Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
				public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
					return (o2.getValue()).compareTo(o1.getValue());
				}
			});

			Map<K, V> result = new LinkedHashMap<K, V>();
			for (Map.Entry<K, V> entry : list) {
				result.put(entry.getKey(), entry.getValue());
			}
			return result;
		}
	}

	private List<Node> getLatents(Graph dag) {
		List<Node> latents = new ArrayList<>();
		for (Node n : dag.getNodes()) {
			if (n.getNodeType() == NodeType.LATENT) {
				latents.add(n);
			}
		}
		return latents;
	}

	//	private static DataSet readInDataSet(Path dataFile) {
	//		DataSet dataSet = null;
	//		DataReader dataReader = new VerticalTabularDiscreteDataReader(dataFile, ',');
	//
	//		try {
	//			dataSet = dataReader.readInData();
	//		} catch (IOException exception) {
	//			String errMsg = String.format("Failed when reading data file '%s'.", dataFile.getFileName());
	//			System.err.println(errMsg);
	//			System.exit(-128);
	//		}
	//
	//		return dataSet;
	//	}


	public static void main(String[] args) throws IOException {
//		int[] variableSize = new int[]{10};
		//		double[] edges = new double[]{2.0, 4.0, 6.0};
		//		double[] lv = new double[]{0.2};
		//		int[] cases = new int[]{200, 1000, 5000};
		//		for (int numCase: cases){
		//			for (int var: variableSize){
		//				for (double epn: edges){
		//					for (double nlv: lv){
		
		// read and process input arguments
		Long seed = 1454147771L;
		String data_path = System.getProperty("user.dir");  
		//"/Users/fattanehjabbari/CCD-Project/CS-BN/dissertation/BSC-Discrete-newTest/";
		double alpha = 0.05, cutoff = 0.5, lower = 0.3, upper = 0.7, edgesPerNode = 2.0, LV = 0.2;;
		int numSim = 1, round = 1, numModels = 100, numBootstrapSamples = 500, numCases = 1000, numVars = 10;
		boolean threshold1 = false, threshold2 = true;

		RBExperiments_sim_old.algorithm = "FCI";
		RBExperiments_sim_old.prior =  0.5;
		RBExperiments_sim_old.completeRules = false;
		RBExperiments_sim_old.depth = -1;

		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "-c":
				numCases = Integer.parseInt(args[i + 1]);
				break;
			case "-v":
				numVars = Integer.parseInt(args[i + 1]);
				break;
			case "-epn":
				edgesPerNode = Double.parseDouble(args[i + 1]);
				break;
			case "-lv":
				LV = Double.parseDouble(args[i + 1]);
				break;
			case "-bs":
				numBootstrapSamples = Integer.parseInt(args[i + 1]);
				break;
			case "-alpha":
				alpha = Double.parseDouble(args[i + 1]);
				break;
			case "-m":
				numModels = Integer.parseInt(args[i + 1]);
				break;
			case "-t1":
				threshold1 = Boolean.parseBoolean(args[i + 1]);
				break;
			case "-t2":
				threshold2 = Boolean.parseBoolean(args[i + 1]);
				break;
			case "-low":
				lower = Double.parseDouble(args[i + 1]);
				break;
			case "-up":
				upper = Double.parseDouble(args[i + 1]);
				break;
			case "-out":
				data_path = args[i + 1];
				break;
			case "-i":
				round = Integer.parseInt(args[i + 1]);
				break;
			}
		}
		RBExperiments_sim_old rbs = new RBExperiments_sim_old();
		rbs.experiment(numModels,alpha, threshold1, threshold2, cutoff, numBootstrapSamples, 
				numVars, edgesPerNode, LV, numCases, numSim, data_path, seed, lower, upper, round);
		//					}
		//				}
		//			}
		//		}
	}


	public void experiment(int numModels ,double alpha, boolean threshold1, boolean threshold2, double cutoff, int numBootstrapSamples,
			int numVars, double edgesPerNode, double latent, int numCases, int numSim, String data_path, 
			long seed, double lower, double upper, int round){

		//		RandomUtil.getInstance().setSeed(seed + 10 * sim);
		//		RandomUtil.getInstance().setSeed(1454147771L);
		RandomUtil.getInstance().setSeed(1454147771L + 10 * round);

		int minCat = 2;
		int maxCat = 4;
		final int numEdges = (int) (numVars * edgesPerNode);
		int numLatents = (int) Math.floor(numVars * latent);

		System.out.println("# nodes: " + numVars + ", # edges: "+ numEdges + ", # numLatents: "+ numLatents + ", # training: " + numCases);

		double[] arrP = new double[numSim], arrR = new double[numSim], adjP = new double[numSim], adjR = new double[numSim], 
				added = new double[numSim], removed = new double[numSim], reoriented = new double[numSim], 
				shdStrict = new double[numSim], shdLenient = new double[numSim], shdAdjacency = new double[numSim];

		double[] arrPI = new double[numSim], arrRI = new double[numSim], adjPI = new double[numSim], adjRI = new double[numSim], 
				addedI = new double[numSim], removedI = new double[numSim], reorientedI = new double[numSim], 
				shdStrictI = new double[numSim], shdLenientI = new double[numSim], shdAdjacencyI = new double[numSim];

		double[] arrPLD = new double[numSim], arrRLD = new double[numSim], adjPLD = new double[numSim], adjRLD = new double[numSim], 
				addedLD = new double[numSim], removedLD = new double[numSim], reorientedLD = new double[numSim], 
				shdStrictLD = new double[numSim], shdLenientLD = new double[numSim], shdAdjacencyLD = new double[numSim];

		double[] arrPD = new double[numSim], arrRD = new double[numSim], adjPD = new double[numSim], adjRD = new double[numSim], 
				addedD = new double[numSim], removedD = new double[numSim], reorientedD = new double[numSim], 
				shdStrictD = new double[numSim], shdLenientD = new double[numSim], shdAdjacencyD = new double[numSim];

		//		PrintStream outlog;

		try {
			File dir = new File(data_path+ "/simulation-" + RBExperiments_sim_old.algorithm + "-depth"+ 
					RBExperiments_sim_old.depth + "-prior"+ RBExperiments_sim_old.prior + "/");
			if (!threshold1){
				cutoff = Double.NaN;
			}
			dir.mkdirs();
			String outputFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-M" + 
					numModels + "-Th1" + threshold1  + "-C" + cutoff + "-BS" + numBootstrapSamples +"-" + 
					RBExperiments_sim_old.algorithm +"-Prior" + RBExperiments_sim_old.prior + "-round" + round + ".csv";

			File file = new File(dir, outputFileName);
			//			File logFile = new File(dir, logFileName);
			if (file.exists() && file.length() != 0){ 
				return;
			}else{
				this.out = new PrintStream(new FileOutputStream(file));
				//				outlog = new PrintStream(new FileOutputStream(logFile));
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}


		// loop over simulations
		for (int s = 0; s < numSim; s++){
			RandomUtil.getInstance().setSeed(1454147771L + 10 * round);

			PrintStream outlog;
			try {
				File dir = new File(data_path+ "/simulation-" + RBExperiments_sim_old.algorithm + 
						"-depth"+ RBExperiments_sim_old.depth +"-prior"+ RBExperiments_sim_old.prior +"/logfiles-V" 
						+ numVars +"-E"+ edgesPerNode +"-L"+ 
						latent + "-N" + numCases + "-M" + numModels + "-BS" + numBootstrapSamples);
				dir.mkdirs();
				if (!threshold1){
					cutoff = Double.NaN;
				}
				String logFileName = "V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases  + "-M" + 
						numModels + "-Th1" + threshold1  + "-C" + cutoff + "-BS" + numBootstrapSamples +"-S" 
						+s + "-" + RBExperiments_sim_old.algorithm  + "-round" + round+".log";
				File logFile = new File(dir, logFileName);
				if (logFile.exists() && logFile.length() != 0){ 
					return;
				}else{
					outlog = new PrintStream(new FileOutputStream(logFile));
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			}

			System.out.println("simulation: " + s);
			outlog.println("simulation: " + s);

			List<Node> vars = createVariables(numVars);

			// generate true BN and its parameters
			Graph dag = GraphUtils.randomGraphRandomForwardEdges(vars, numLatents, numEdges, 15, 10, 10, false, true);
			System.out.println("Latent variables: " + getLatents(dag));
			outlog.println("Latent variables: " + getLatents(dag));

			System.out.println("generating pm ...");

			BayesPm pm = new BayesPm(dag, minCat, maxCat);
			System.out.println("generating im ...");

			MlBayesIm im = new MlBayesIm(pm, MlBayesIm.RANDOM);

			System.out.println("simulating data ...");
			// simulate train and test data from BN
			DataSet fullTrainData = im.simulateData(numCases, true);

			// get the observed part of the data only
			DataSet data = DataUtils.restrictToMeasured(fullTrainData);

			// get the true underlying PAG
			final DagToPag2 dagToPag = new DagToPag2(dag);
			dagToPag.setCompleteRuleSetUsed(RBExperiments_sim_old.completeRules);
			Graph PAG_True = dagToPag.convert();
			PAG_True = GraphUtils.replaceNodes(PAG_True, data.getVariables());
			outlog.println("PAG_True: " +PAG_True);

			// run RFCI to get a PAG using chi-squared test
			long start = System.currentTimeMillis();
			Graph rfciPag = runPagCs(data, alpha, PAG_True);
			long RfciTime = System.currentTimeMillis() - start;
			System.out.println("FCI done!");

			// run RFCI-BSC (RB) search using BSC test and obtain constraints that
			// are queried during the search
			List<Graph> bscPags = new ArrayList<Graph>();
			start = System.currentTimeMillis();
			IndTestProbabilisticBDeu testBSC = runRB(data, bscPags, numModels, threshold1, PAG_True);
			long BscRfciTime = System.currentTimeMillis() - start;
			Map<IndependenceFact, Double> H = testBSC.getH();
			System.out.println("FB (FCI-BSC) done!");
			//
			// create empirical data for constraints
			start = System.currentTimeMillis();
			DataSet depData = createDepDataFiltering(H, data, numBootstrapSamples, threshold2, lower, upper);
			System.out.println("H Size:" + H.size());
			System.out.println("DepData(row,col):" + depData.getNumRows() + "," + depData.getNumColumns());

			outlog.println("H Size:" + H.size());
			outlog.println("DepData(row,col):" + depData.getNumRows() + "," + depData.getNumColumns());

			// learn structure of constraints using empirical data
			Graph depPattern = runFGS(depData);
			Graph estDepBN = SearchGraphUtils.dagFromPattern(depPattern);

			System.out.println("estDepBN: " + estDepBN.getEdges());

			outlog.println("DepGraph(nodes,edges):" + estDepBN.getNumNodes() + "," + estDepBN.getNumEdges());
			for (Edge e: estDepBN.getEdges())
				outlog.println(e);

			//			System.out.println("Dependency graph done!");

			// estimate parameters of the graph learned for constraints
			BayesPm pmHat = new BayesPm(estDepBN, 2, 2);
			DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(pmHat, 0.5);
			BayesIm imHat = DirichletEstimator.estimate(prior, depData);
			Long BscdTime = System.currentTimeMillis() - start;
			System.out.println("Dependency BN_Param done");


			Map<NodePair, GraphParameter> localBNs =  learnLocalBNs(data.getVariables(), depData, H, lower, upper);
			for (NodePair npair: localBNs.keySet()){
				GraphParameter ab_g=localBNs.get(npair);
				//				outlog.println("LocalDepGraph(nodes,edges):" + ab_g.g.getNumNodes() + "," + ab_g.g.getNumEdges());
				outlog.println("localDepGraph (pairs of nodes):");
				if (ab_g.g.getNumEdges() > 0){		
					outlog.println("nodepair: " + npair.n_a + ", " + npair.n_b);
					for (Edge e: ab_g.g.getEdges())
						outlog.println(e);
				}
			}


			System.out.println("Local dependency graphs done!");

			// compute scores of graphs that are output by RB search using BSC-I and
			// BSC-D methods
			start = System.currentTimeMillis();
			allScores lnProbs = getLnProbsAll(bscPags, H, data, imHat, estDepBN, localBNs, outlog);
			Long mutualTime = (System.currentTimeMillis() - start) / 2;

			// normalize the scores
			start = System.currentTimeMillis();
			Map<Graph, Double> normalizedLocalDep = normalProbs(lnProbs.LnBSCLD);
			Long ldTime = System.currentTimeMillis() - start;

			// normalize the scores
			start = System.currentTimeMillis();
			Map<Graph, Double> normalizedDep = normalProbs(lnProbs.LnBSCD);
			Long dTime = System.currentTimeMillis() - start;

			start = System.currentTimeMillis();
			Map<Graph, Double> normalizedInd = normalProbs(lnProbs.LnBSCI);
			Long iTime = System.currentTimeMillis() - start;

			// get the most probable PAG using each scoring method
			normalizedLocalDep = MapUtil.sortByValue(normalizedLocalDep);
			Graph maxBNLD = normalizedLocalDep.keySet().iterator().next();
			outlog.println("maxBNLD Prob: " + normalizedLocalDep.get(maxBNLD));
			outlog.println("maxBNLD: "+ maxBNLD);

			normalizedDep = MapUtil.sortByValue(normalizedDep);
			Graph maxBND = normalizedDep.keySet().iterator().next();
			outlog.println("maxBND:  Prob: " + normalizedDep.get(maxBND));
			outlog.println("maxBND: " + maxBND);

			normalizedInd = MapUtil.sortByValue(normalizedInd);
			Graph maxBNI = normalizedInd.keySet().iterator().next();
			outlog.println("maxBNI Prob: " + normalizedInd.get(maxBNI));
			outlog.println("Estimated Graph: " + maxBNI);
			outlog.println();

			for (int ind_g =0 ; ind_g < bscPags.size(); ind_g ++){	
				outlog.println("S_BSCI: " + lnProbs.LnBSCI.get(bscPags.get(ind_g)));
				outlog.println("P_BSCI: " + normalizedInd.get(bscPags.get(ind_g)));

				outlog.println("S_BSCLD: " + lnProbs.LnBSCLD.get(bscPags.get(ind_g)));
				outlog.println("P_BSCLD: " + normalizedLocalDep.get(bscPags.get(ind_g)));

				outlog.println("S_BSCD: " + lnProbs.LnBSCD.get(bscPags.get(ind_g)));
				outlog.println("P_BSCD: " + normalizedDep.get(bscPags.get(ind_g)));
				outlog.println("Graph_i:" + bscPags.get(ind_g));

			}
			outlog.println("------------------------------------------");

			ArrowConfusion congI = new ArrowConfusion(PAG_True, GraphUtils.replaceNodes(maxBNI, PAG_True.getNodes()));
			AdjacencyConfusion conAdjGI = new AdjacencyConfusion(PAG_True, GraphUtils.replaceNodes(maxBNI, PAG_True.getNodes()));

			double denP = (congI.getArrowsTp()+congI.getArrowsFp());
			double denR = (congI.getArrowsTp()+congI.getArrowsFn());
			if (denP == 0.0 && denR == 0.0){
				arrPI[s] += 1.0;
				arrRI[s] += 1.0;
			}
			if (denP != 0.0){
				arrPI[s] += (congI.getArrowsTp() / denP);
			}
			if (denR != 0.0){
				arrRI[s] += (congI.getArrowsTp() / denR);
			}


			denP = (conAdjGI.getAdjTp() + conAdjGI.getAdjFp());
			denR = (conAdjGI.getAdjTp() + conAdjGI.getAdjFn());
			if (denP == 0.0 && denR == 0.0){
				adjPI[s] += 1.0;
				adjRI[s] += 1.0;
			}
			if (denP != 0.0){
				adjPI[s] += (conAdjGI.getAdjTp() / denP);
			}
			if (denR != 0.0){
				adjRI[s] += (conAdjGI.getAdjTp() / denR);
			}

			ArrowConfusion congLD= new ArrowConfusion(PAG_True, GraphUtils.replaceNodes(maxBNLD, PAG_True.getNodes()));
			AdjacencyConfusion conAdjGLD= new AdjacencyConfusion(PAG_True, GraphUtils.replaceNodes(maxBNLD, PAG_True.getNodes()));

			denP = (congLD.getArrowsTp()+congLD.getArrowsFp());
			denR = (congLD.getArrowsTp()+congLD.getArrowsFn());
			if (denP == 0.0 && denR == 0.0){
				arrPLD[s] += 1.0;
				arrRLD[s] += 1.0;
			}
			if (denP != 0.0){
				arrPLD[s] += (congLD.getArrowsTp() / denP);
			}
			if (denR != 0.0){
				arrRLD[s] += (congLD.getArrowsTp() / denR);
			}


			denP = (conAdjGLD.getAdjTp() + conAdjGLD.getAdjFp());
			denR = (conAdjGLD.getAdjTp() + conAdjGLD.getAdjFn());
			if (denP == 0.0 && denR == 0.0){
				adjPLD[s] += 1.0;
				adjRLD[s] += 1.0;
			}
			if (denP != 0.0){
				adjPLD[s] += (conAdjGLD.getAdjTp() / denP);
			}
			if (denR != 0.0){
				adjRLD[s] += (conAdjGLD.getAdjTp() / denR);
			}

			ArrowConfusion congD = new ArrowConfusion(PAG_True, GraphUtils.replaceNodes(maxBND, PAG_True.getNodes()));
			AdjacencyConfusion conAdjGD = new AdjacencyConfusion(PAG_True, GraphUtils.replaceNodes(maxBND, PAG_True.getNodes()));

			denP = (congD.getArrowsTp()+congD.getArrowsFp());
			denR = (congD.getArrowsTp()+congD.getArrowsFn());
			if (denP == 0.0 && denR == 0.0){
				arrPD[s] += 1.0;
				arrRD[s] += 1.0;
			}
			if (denP != 0.0){
				arrPD[s] += (congD.getArrowsTp() / denP);
			}

			if (denR != 0.0){
				arrRD[s] += (congD.getArrowsTp() / denR);
			}


			denP = (conAdjGD.getAdjTp() + conAdjGD.getAdjFp());
			denR = (conAdjGD.getAdjTp() + conAdjGD.getAdjFn());
			if (denP == 0.0 && denR == 0.0){
				adjPD[s] += 1.0;
				adjRD[s] += 1.0;
			}
			if (denP != 0.0){
				adjPD[s] += (conAdjGD.getAdjTp() / denP);
			}
			if (denR != 0.0){
				adjRD[s] += (conAdjGD.getAdjTp() / denR);
			}


			ArrowConfusion cong = new ArrowConfusion(PAG_True, GraphUtils.replaceNodes(rfciPag, PAG_True.getNodes()));
			AdjacencyConfusion conAdjG = new AdjacencyConfusion(PAG_True, GraphUtils.replaceNodes(rfciPag, PAG_True.getNodes()));

			System.out.println();
			// population model evaluation
			denP = (cong.getArrowsTp() + cong.getArrowsFp());
			denR = (cong.getArrowsTp() + cong.getArrowsFn());
			if (denP == 0.0 && denR == 0.0){
				arrP[s] += 1.0;
				arrR[s] += 1.0;
			}
			if (denP != 0.0){
				arrP[s] = (cong.getArrowsTp() / denP);
			}
			if (denR != 0.0){
				arrR[s] = (cong.getArrowsTp() / denR);
			}

			denP = (conAdjG.getAdjTp() + conAdjG.getAdjFp());
			denR = (conAdjG.getAdjTp() + conAdjG.getAdjFn());
			if (denP == 0.0 && denR == 0.0){
				adjP[s] += 1.0;
				adjR[s] += 1.0;
			}
			if (denP != 0.0){
				adjP[s] = (conAdjG.getAdjTp() / denP);
			}
			if (denR != 0.0){
				adjR[s] = (conAdjG.getAdjTp() / denR);
			}
			GraphUtils.GraphComparison cmpI = SearchGraphUtils.getGraphComparison(maxBNI, PAG_True, true);
			GraphUtils.GraphComparison cmpLD = SearchGraphUtils.getGraphComparison(maxBNLD, PAG_True, true);
			GraphUtils.GraphComparison cmpD = SearchGraphUtils.getGraphComparison(maxBND, PAG_True, true);
			GraphUtils.GraphComparison cmpP = SearchGraphUtils.getGraphComparison(rfciPag, PAG_True, true);

			addedI[s] = cmpI.getEdgesAdded().size();
			removedI[s] = cmpI.getEdgesRemoved().size();
			reorientedI[s] = cmpI.getEdgesReorientedTo().size();
			shdStrictI[s] = cmpI.getShdStrict();
			shdLenientI[s] = cmpI.getShdLenient();
			shdAdjacencyI[s] = cmpI.getEdgesAdded().size() + cmpI.getEdgesRemoved().size();

			addedLD[s] = cmpLD.getEdgesAdded().size();
			removedLD[s] = cmpLD.getEdgesRemoved().size();
			reorientedLD[s] = cmpLD.getEdgesReorientedTo().size();
			shdStrictLD[s] = cmpLD.getShdStrict();
			shdLenientLD[s] = cmpLD.getShdLenient();
			shdAdjacencyLD[s] = cmpLD.getEdgesAdded().size() + cmpLD.getEdgesRemoved().size();


			addedD[s] = cmpD.getEdgesAdded().size();
			removedD[s] = cmpD.getEdgesRemoved().size();
			reorientedD[s] = cmpD.getEdgesReorientedTo().size();
			shdStrictD[s] = cmpD.getShdStrict();
			shdLenientD[s] = cmpD.getShdLenient();
			shdAdjacencyD[s] = cmpD.getEdgesAdded().size() + cmpD.getEdgesRemoved().size();


			added[s] = cmpP.getEdgesAdded().size();
			removed[s] = cmpP.getEdgesRemoved().size();
			reoriented[s] = cmpP.getEdgesReorientedTo().size();
			shdStrict[s] = cmpP.getShdStrict();
			shdLenient[s] = cmpP.getShdLenient();
			shdAdjacency[s] = cmpP.getEdgesAdded().size() + cmpP.getEdgesRemoved().size();
		}

		printRes(this.out, "POP BSC-I", numSim, arrPI, arrRI, adjPI, adjRI, addedI, removedI, reorientedI, shdStrictI, shdLenientI, shdAdjacencyI);
		printRes(this.out, "POP BSC-D", numSim, arrPD, arrRD, adjPD, adjRD, addedD, removedD, reorientedD, shdStrictD, shdLenientD, shdAdjacencyD);
		printRes(this.out, "POP BSC-LD", numSim, arrPLD, arrRLD, adjPLD, adjRLD, addedLD, removedLD, reorientedLD, shdStrictLD, shdLenientLD, shdAdjacencyLD);
		printRes(this.out,"POP Chi2", numSim, arrP, arrR, adjP, adjR, added, removed, reoriented, shdStrict, shdLenient, shdAdjacency);
		this.out.close();
		System.out.println("----------------------");

	}
	private Map<NodePair, GraphParameter> learnLocalBNs(List<Node> variables, DataSet depData, Map<IndependenceFact, Double> H, double lower, double upper){

		Map<NodePair, GraphParameter> localBNs = new HashMap<>();

		for (int i = 0; i < variables.size(); i++) {
			for (int j = i + 1; j < variables.size(); j++) {

				// get a pair of nodes (a,b)
				NodePair ab = new NodePair(variables.get(i).getName(), variables.get(j).getName());


				// obtain the tests that are about (a,b)	
				Map<IndependenceFact, Double> H_ab = groupHbyNodePair (ab, H, lower, upper);

				// if there are any relevant tests to pair (a,b), then
				if (H_ab.size() > 1){
					//            		System.out.println(ab.print(ab));
					//            		System.out.println("H_ab: " + H_ab.size());

					// 1. obtain a part of depData that is about the relevant tests
					DataSet depData_ab = getDataSubset (depData, H_ab);

					// 2. learn a BN for (a,b) group
					Graph depPattern_ab = runFGS(depData_ab);
					Graph estDepBN_ab = SearchGraphUtils.dagFromPattern(depPattern_ab);

					// 3. estimate parameters of the graph learned for (a,b) group
					BayesPm pmHat_ab = new BayesPm(estDepBN_ab, 2, 2);
					DirichletBayesIm prior_ab = DirichletBayesIm.symmetricDirichletIm(pmHat_ab, 0.5);
					BayesIm imHat_ab = DirichletEstimator.estimate(prior_ab, depData_ab);
					GraphParameter g_theta= new GraphParameter(estDepBN_ab,imHat_ab);
					//					g_theta.put(estDepBN_ab, imHat_ab);
					localBNs.put(ab, g_theta);
				}	
			}
		}
		return localBNs;
	}

	private Map<IndependenceFact, Double> groupHbyNodePair(NodePair ab, Map<IndependenceFact, Double> H, double lower, double upper){
		Map<IndependenceFact, Double> H_ab = new HashMap<>();
		for (IndependenceFact f: H.keySet()){
			if ( (f.getX().getName().equals(ab.n_a) && f.getY().getName().equals(ab.n_b)) || 
					(f.getX().getName().equals(ab.n_b) && f.getY().getName().equals(ab.n_a)) ){
				if (H.get(f) >= lower && H.get(f) <= upper ){
					H_ab.put(f, H.get(f));
				}
			}
		}
		return H_ab;
	}
	private DataSet getDataSubset (DataSet depData, Map<IndependenceFact, Double> H_ab){
		List<Node> vars = new ArrayList<>();
		for (IndependenceFact f : H_ab.keySet()) {

			vars.add(depData.getVariable(f.toString()));
		}

		return depData.subsetColumns(vars);
	}
	private List<Node> createVariables(int numVars) {
		// create variables
		List<Node> vars = new ArrayList<>();
		for (int i = 0; i < numVars; i++) {
			vars.add(new DiscreteVariable("X" + i));
		}
		return vars;
	}
	//	public void experiment_ECML(String modelName, int numCases, int numModels, int numBootstrapSamples, double alpha,
	//			String filePath, int round) {
	//		// 32827167123L
	//		Long seed = 878376L;
	//		RandomUtil.getInstance().setSeed(seed);
	//		PrintStream out;
	//
	//		// get the Bayesian network (graph and parameters) of the given model
	//		BayesIm im = getBayesIM(modelName);
	//		System.out.println("im:" + im);
	//		BayesPm pm = im.getBayesPm();
	//		Graph dag = pm.getDag();
	//
	//		// set the "numLatentConfounders" percentage of variables to be latent
	//		int numVars = im.getNumNodes();
	//		int LV = (int) Math.floor(numLatentConfounders * numVars);
	//		GraphUtils.fixLatents4(LV, dag);
	//		System.out.println("Variables set to be latent:" +getLatents(dag));
	//
	//		// create output directory and files
	//		filePath = filePath + "/" + modelName + "-Vars" + dag.getNumNodes() + "-Edges" + dag.getNumEdges() + "-H"
	//				+ numLatentConfounders + "-Cases" + numCases + "-numModels" + numModels + "-BS" + numBootstrapSamples;
	//		try {
	//			File dir = new File(filePath);
	//			dir.mkdirs();
	//			File file = new File(dir,
	//					"Results-" + modelName + "-Vars" + dag.getNumNodes() + "-Edges" + dag.getNumEdges() + "-H"
	//							+ numLatentConfounders + "-Cases" + numCases + "-numModels" + numModels + "-BS"
	//							+ numBootstrapSamples + "-" + round + ".txt");
	//			if (!file.exists() || file.length() == 0) {
	//				out = new PrintStream(new FileOutputStream(file));
	//			} else {
	//				return;
	//			}
	//		} catch (Exception e) {
	//			throw new RuntimeException(e);
	//		}
	//
	//		// simulate data from instantiated model
	//		DataSet fullData = im.simulateData(numCases, round * 1000000 + 71512, true);
	//		fullData = refineData(fullData);
	//		DataSet data = DataUtils.restrictToMeasured(fullData);
	//
	//		// get the true underlying PAG
	//		final DagToPag2 dagToPag = new DagToPag2(dag);
	//		dagToPag.setCompleteRuleSetUsed(false);
	//		Graph PAG_True = dagToPag.convert();
	//		PAG_True = GraphUtils.replaceNodes(PAG_True, data.getVariables());
	//
	//		// run RFCI to get a PAG using chi-squared test
	//		long start = System.currentTimeMillis();
	//		Graph rfciPag = runPagCs(data, alpha);
	//		long RfciTime = System.currentTimeMillis() - start;
	//		System.out.println("RFCI done!");
	//
	//		// run RFCI-BSC (RB) search using BSC test and obtain constraints that
	//		// are queried during the search
	//		List<Graph> bscPags = new ArrayList<Graph>();
	//		start = System.currentTimeMillis();
	//		IndTestProbabilisticBDeu testBSC = runRB(data, bscPags, numModels, threshold1);
	//		long BscRfciTime = System.currentTimeMillis() - start;
	//		Map<IndependenceFact, Double> H = testBSC.getH();
	//		//		out.println("H Size:" + H.size());
	//		System.out.println("RB (RFCI-BSC) done!");
	//		//
	//		// create empirical data for constraints
	//		start = System.currentTimeMillis();
	//		DataSet depData = createDepDataFiltering(H, data, numBootstrapSamples, threshold2, lower, upper);
	//		out.println("DepData(row,col):" + depData.getNumRows() + "," + depData.getNumColumns());
	//		System.out.println("Dep data creation done!");
	//
	//		// learn structure of constraints using empirical data
	//		Graph depPattern = runFGS(depData);
	//		Graph estDepBN = SearchGraphUtils.dagFromPattern(depPattern);
	//		System.out.println("estDepBN: " + estDepBN.getEdges());
	//		out.println("DepGraph(nodes,edges):" + estDepBN.getNumNodes() + "," + estDepBN.getNumEdges());
	//		System.out.println("Dependency graph done!");
	//
	//		// estimate parameters of the graph learned for constraints
	//		BayesPm pmHat = new BayesPm(estDepBN, 2, 2);
	//		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(pmHat, 0.5);
	//		BayesIm imHat = DirichletEstimator.estimate(prior, depData);
	//		Long BscdTime = System.currentTimeMillis() - start;
	//		System.out.println("Dependency BN_Param done");
	//
	//		// compute scores of graphs that are output by RB search using BSC-I and
	//		// BSC-D methods
	//		start = System.currentTimeMillis();
	//		allScores lnProbs = getLnProbsAll(bscPags, H, data, imHat, estDepBN);
	//		Long mutualTime = (System.currentTimeMillis() - start) / 2;
	//
	//		// normalize the scores
	//		start = System.currentTimeMillis();
	//		Map<Graph, Double> normalizedDep = normalProbs(lnProbs.LnBSCD);
	//		Long dTime = System.currentTimeMillis() - start;
	//
	//		start = System.currentTimeMillis();
	//		Map<Graph, Double> normalizedInd = normalProbs(lnProbs.LnBSCI);
	//		Long iTime = System.currentTimeMillis() - start;
	//
	//		// get the most probable PAG using each scoring method
	//		normalizedDep = MapUtil.sortByValue(normalizedDep);
	//		Graph maxBND = normalizedDep.keySet().iterator().next();
	//		//		Graph maxBND = bscPags.get(0);
	//
	//		normalizedInd = MapUtil.sortByValue(normalizedInd);
	//		Graph maxBNI = normalizedInd.keySet().iterator().next();
	//		//		Graph maxBNI = bscPags.get(0);
	//
	//		// summarize and write the results into output files
	//		out.println("*** RFCI time (sec):" + (RfciTime / 1000));
	//		summarize(rfciPag, PAG_True, out);
	//
	//		out.println("\n*** RB-I time (sec):" + BscRfciTime);// + mutualTime + iTime) / 1000));
	//		summarize(maxBNI, PAG_True, out);
	//		//
	//		out.println("\n*** RB-D time (sec):" + BscRfciTime);// + BscdTime + mutualTime + dTime) / 1000));
	//		summarize(maxBND, PAG_True, out);
	//		//
	//		out.println("P(maxBNI): \n" + 1.0);//normalizedInd.get(maxBNI));
	//		out.println("P(maxBND): \n" + 1.0);// normalizedDep.get(maxBND));
	//		//		out.println("------------------------------------------");
	//		//		out.println(normalizedInd.values());
	//		//		out.println("------------------------------------------");
	//		//		out.println(normalizedDep.values());
	//		out.println("------------------------------------------");
	//		out.println("PAG_True: \n" + PAG_True);
	//		out.println("------------------------------------------");
	//		out.println("Rfci: \n" + rfciPag);
	//		out.println("------------------------------------------");
	//		out.println("RB-I: \n" + maxBNI);
	//		out.println("------------------------------------------");
	//		out.println("RB-D: \n" + maxBND);
	//
	//		out.close();
	//
	//	}

	/*public void experiment2(String modelName, int numCases, int numModels, int numBootstrapSamples, double alpha,
			double numLatentConfounders, boolean threshold1, boolean threshold2, double lower, double upper,
			String filePath, int round) {
		// 32827167123L
		modelName = "MCBN1";
		Long seed = 32827167123L;
		RandomUtil.getInstance().setSeed(seed);
		PrintStream out;

		// Path to data file

		// create a Bayesian network (dag) and its parameters
		Graph dag = makeSimpleDAG(0);
		BayesPm pm = new BayesPm(dag, 2, 2);
		BayesIm im = new MlBayesIm(pm, MlBayesIm.MANUAL);
		im = initializeIM(im);
		System.out.println("MCBN1: " + dag);


		// set the "numLatentConfounders" percentage of variables to be latent
		int numVars = im.getNumNodes();
		int LV = (int) Math.floor(numLatentConfounders * numVars);
		//		GraphUtils.fixLatents4(LV, dag);
		System.out.println("Variables set to be latent:" +getLatents(dag));
		Path dataFile = Paths.get("/Users/fattanehjabbari/Downloads/", "mcbn1.txt");
		//		DataSet fullData = readInDataSet(dataFile);
		DataSet fullData = im.simulateData(numCases, round * 1000000 + 71512, true);
		DataSet data = DataUtils.restrictToMeasured(fullData);

		// create output directory and files
		filePath = filePath + "/" + modelName + "-Vars" + dag.getNumNodes() + "-Edges" + dag.getNumEdges() + "-H"
				+ numLatentConfounders + "-Cases" + numCases + "-numModels" + numModels + "-BS" + numBootstrapSamples;
		try {
			File dir = new File(filePath);
			dir.mkdirs();
			File file = new File(dir,
					"Results-" + modelName + "-Vars" + dag.getNumNodes() + "-Edges" + dag.getNumEdges() + "-H"
							+ numLatentConfounders + "-Cases" + numCases + "-numModels" + numModels + "-BS"
							+ numBootstrapSamples + "-" + round + ".txt");
			if (!file.exists() || file.length() == 0) {
				out = new PrintStream(new FileOutputStream(file));
			} else {
				return;
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

		// get the true underlying PAG
		final DagToPag dagToPag = new DagToPag(dag);
		dagToPag.setCompleteRuleSetUsed(false);
		Graph PAG_True = dagToPag.convert();
		PAG_True = GraphUtils.replaceNodes(PAG_True, data.getVariables());

		// run RFCI to get a PAG using chi-squared test
		long start = System.currentTimeMillis();
		Graph rfciPag = runPagCs(data, alpha);
		long RfciTime = System.currentTimeMillis() - start;
		System.out.println("RFCI done!");

		// run RFCI-BSC (RB) search using BSC test and obtain constraints that
		// are queried during the search
		List<Graph> bscPags = new ArrayList<Graph>();
		start = System.currentTimeMillis();
		IndTestProbabilistic testBSC = runRB(data, bscPags, numModels, threshold1);
		long BscRfciTime = System.currentTimeMillis() - start;
		Map<IndependenceFact, Double> H = testBSC.getH();
		//		out.println("H Size:" + H.size());
		System.out.println("RB (RFCI-BSC) done!");

		// create empirical data for constraints
		start = System.currentTimeMillis();
		DataSet depData = createDepDataFiltering(H, data, numBootstrapSamples, threshold2, lower, upper);
		//		out.println("DepData(row,col):" + depData.getNumRows() + "," + depData.getNumColumns());
		System.out.println("Dep data creation done!");

		// learn structure of constraints using empirical data => constraint meta data
		Graph depPattern = runFGS(depData);
		Graph estDepBN = SearchGraphUtils.dagFromPattern(depPattern);
		System.out.println("estDepBN: " + estDepBN.getEdges());
		//		out.println("DepGraph(nodes,edges):" + estDepBN.getNumNodes() + "," + estDepBN.getNumEdges());
		System.out.println("Dependency graph done!");

		// estimate parameters of the graph learned for constraints
		BayesPm pmHat = new BayesPm(estDepBN, 2, 2);
		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(pmHat, 0.5);
		BayesIm imHat = DirichletEstimator.estimate(prior, depData);
		Long BscdTime = System.currentTimeMillis() - start;
		System.out.println("Dependency BN_Param done");

		// compute scores of graphs that are output by RB search using BSC-I and
		// BSC-D methods
		start = System.currentTimeMillis();
		allScores lnProbs = getLnProbsAll(bscPags, H, data, imHat, estDepBN);
		Long mutualTime = (System.currentTimeMillis() - start) / 2;

		// normalize the scores
		start = System.currentTimeMillis();
		Map<Graph, Double> normalizedDep = normalProbs(lnProbs.LnBSCD);
		Long dTime = System.currentTimeMillis() - start;

		start = System.currentTimeMillis();
		Map<Graph, Double> normalizedInd = normalProbs(lnProbs.LnBSCI);
		Long iTime = System.currentTimeMillis() - start;

		// get the most probable PAG using each scoring method
		normalizedDep = MapUtil.sortByValue(normalizedDep);
		Graph maxBND = normalizedDep.keySet().iterator().next();

		normalizedInd = MapUtil.sortByValue(normalizedInd);
		Graph maxBNI = normalizedInd.keySet().iterator().next();

		// summarize and write the results into output files
		out.println("*** RFCI time (sec):" + (RfciTime / 1000));
		summarize(rfciPag, PAG_True, out);

		out.println("\n*** RB-I time (sec):" + ((BscRfciTime + mutualTime + iTime) / 1000));
		summarize(maxBNI, PAG_True, out);

		out.println("\n*** RB-D time (sec):" + ((BscRfciTime + BscdTime + mutualTime + dTime) / 1000));
		summarize(maxBND, PAG_True, out);

		out.println("P(maxBNI): \n" + normalizedInd.get(maxBNI));
		out.println("P(maxBND): \n" + normalizedDep.get(maxBND));
		out.println("------------------------------------------");
		out.println(normalizedInd.values());
		out.println("------------------------------------------");
		out.println(normalizedDep.values());
		out.println("------------------------------------------");
		out.println("PAG_True: \n" + PAG_True);
		out.println("------------------------------------------");
		out.println("Rfci: \n" + rfciPag);
		out.println("------------------------------------------");
		out.println("RB-I: \n" + maxBNI);
		out.println("------------------------------------------");
		out.println("RB-D: \n" + maxBND);

		out.close();

	}
	 */

	//	private DataSet refineData(DataSet fullData) {
	//		for (int c = 0; c < fullData.getNumColumns(); c++) {
	//			for (int r = 0; r < fullData.getNumRows(); r++) {
	//				if (fullData.getInt(r, c) < 0) {
	//					fullData.setInt(r, c, 0);
	//				}
	//			}
	//		}
	//
	//		return fullData;
	//	}


	private void printRes(PrintStream out, String alg, int numSim, double[] arrPI, double[] arrRI, 
			double[] adjPI, double[] adjRI, 
			double[] addedI, double[] removedI, double[] reorientedI, 
			double[] shdStrictI, double[] shdLenientI, double[] shdAdjacencyI){

		NumberFormat nf = new DecimalFormat("0.00");
		//			NumberFormat smallNf = new DecimalFormat("0.00E0");

		TextTable table = new TextTable(numSim+2, 8);
		table.setTabDelimited(true);
		String header = ", adj_P, adj_R, arr_P, arr_R, added, removed, reoriented, SHD_S, SHD_L, SHD_A";
		table.setToken(0, 0, alg);
		table.setToken(0, 1, header);
		double arrP = 0.0, arrR = 0.0, adjP = 0.0, adjR = 0.0,
				added = 0.0, removed = 0.0, reoriented = 0.0, shdStrict = 0.0, shdLenient =0.0, shdAdjacency = 0.0;
		for (int i = 0; i < numSim; i++){
			String res = "," + nf.format(adjPI[i]) + "," + nf.format(adjRI[i])
			+ "," + nf.format(arrPI[i]) + "," + nf.format(arrRI[i])
			+ "," + nf.format(addedI[i]) + "," + nf.format(removedI[i]) + "," + nf.format(reorientedI[i])
			+ "," + nf.format(shdStrictI[i])+","+nf.format(shdLenientI[i])+ "," + nf.format(shdAdjacencyI[i]) ;
			table.setToken(i+1, 0, ""+(i+1));
			table.setToken(i+1, 1, res);

			arrP += arrPI[i];
			arrR += arrRI[i];
			adjP += adjPI[i];
			adjR += adjRI[i];
			added += addedI[i];
			removed += removedI[i];
			reoriented += reorientedI[i];
			shdStrict += shdStrictI[i];
			shdLenient += shdLenientI[i];
			shdAdjacency += shdAdjacencyI[i];
		}
		String res =  ","+nf.format(adjP/numSim)+","+nf.format(adjR/numSim)+","+
				nf.format(arrP/numSim)+","+nf.format(arrR/numSim)+","+
				nf.format(added/numSim)+","+
				nf.format(removed/numSim)+","+
				nf.format(reoriented/numSim)+","+
				nf.format(shdStrict/numSim)+","+nf.format(shdLenient/numSim)+","+nf.format(shdAdjacency/numSim);

		table.setToken(numSim+1, 0, "avg");
		table.setToken(numSim+1, 1, res);
		out.println(table);
		System.out.println(table);		
	}


	//	private void summarize(Graph graph, Graph trueGraph, PrintStream out) {
	//		out.flush();
	//		ArrayList<Comparison.TableColumn> tableColumns = new ArrayList<>();
	//
	//		tableColumns.add(Comparison.TableColumn.AhdCor);
	//		tableColumns.add(Comparison.TableColumn.AhdFp);
	//		tableColumns.add(Comparison.TableColumn.AhdFn);
	//		tableColumns.add(Comparison.TableColumn.AhdPrec);
	//		tableColumns.add(Comparison.TableColumn.AhdRec);
	//
	//		tableColumns.add(Comparison.TableColumn.AdjCor);
	//		tableColumns.add(Comparison.TableColumn.AdjFp);
	//		tableColumns.add(Comparison.TableColumn.AdjFn);
	//		tableColumns.add(Comparison.TableColumn.AdjPrec);
	//		tableColumns.add(Comparison.TableColumn.AdjRec);
	//
	//		tableColumns.add(Comparison.TableColumn.SHD);
	//
	//		GraphUtils.GraphComparison comparison = SearchGraphUtils.getGraphComparison3(graph, trueGraph, null);
	//
	//		List<Node> variables = new ArrayList<>();
	//		for (TableColumn column : tableColumns) {
	//			variables.add(new ContinuousVariable(column.toString()));
	//		}
	//
	//		DataSet dataSet = new BoxDataSet(new DoubleDataBox(0, variables.size()), variables);
	//		dataSet.setNumberFormat(new DecimalFormat("0"));
	//
	//		int newRow = dataSet.getNumRows();
	//
	//		if (tableColumns.contains(TableColumn.AdjCor)) {
	//			dataSet.setDouble(newRow, tableColumns.indexOf(TableColumn.AdjCor), comparison.getAdjCor());
	//		}
	//
	//		if (tableColumns.contains(TableColumn.AdjFn)) {
	//			dataSet.setDouble(newRow, tableColumns.indexOf(TableColumn.AdjFn), comparison.getAdjFn());
	//		}
	//
	//		if (tableColumns.contains(TableColumn.AdjFp)) {
	//			dataSet.setDouble(newRow, tableColumns.indexOf(TableColumn.AdjFp), comparison.getAdjFp());
	//		}
	//
	//		if (tableColumns.contains(TableColumn.AdjPrec)) {
	//			dataSet.setDouble(newRow, tableColumns.indexOf(TableColumn.AdjPrec), comparison.getAdjPrec());
	//		}
	//
	//		if (tableColumns.contains(TableColumn.AdjRec)) {
	//			dataSet.setDouble(newRow, tableColumns.indexOf(TableColumn.AdjRec), comparison.getAdjRec());
	//		}
	//
	//		if (tableColumns.contains(TableColumn.AhdCor)) {
	//			dataSet.setDouble(newRow, tableColumns.indexOf(TableColumn.AhdCor), comparison.getAhdCor());
	//		}
	//
	//		if (tableColumns.contains(TableColumn.AhdFn)) {
	//			dataSet.setDouble(newRow, tableColumns.indexOf(TableColumn.AhdFn), comparison.getAhdFn());
	//		}
	//
	//		if (tableColumns.contains(TableColumn.AhdFp)) {
	//			dataSet.setDouble(newRow, tableColumns.indexOf(TableColumn.AhdFp), comparison.getAhdFp());
	//		}
	//
	//		if (tableColumns.contains(TableColumn.AhdPrec)) {
	//			dataSet.setDouble(newRow, tableColumns.indexOf(TableColumn.AhdPrec), comparison.getAhdPrec());
	//		}
	//
	//		if (tableColumns.contains(TableColumn.AhdRec)) {
	//			dataSet.setDouble(newRow, tableColumns.indexOf(TableColumn.AhdRec), comparison.getAhdRec());
	//		}
	//
	//		if (tableColumns.contains(TableColumn.SHD)) {
	//			dataSet.setDouble(newRow, tableColumns.indexOf(TableColumn.SHD), comparison.getShd());
	//		}
	//
	//		int[] cols = new int[tableColumns.size()];
	//		for (int i = 0; i < cols.length; i++) {
	//			cols[i] = i;
	//		}
	//
	//		//		out.println(getTextTable(dataSet, cols, new DecimalFormat("0.00")));
	//		out.println(MisclassificationUtils.edgeMisclassifications(graph, trueGraph));
	//		//		printCorrectArrows(graph, trueGraph, out);
	//		//		printCorrectTails(graph, trueGraph, out);
	//		int SHDAdj = comparison.getEdgesAdded().size() + comparison.getEdgesRemoved().size();
	//		int diffEdgePoint = comparison.getShd() - SHDAdj * 2;
	//		out.println("# missing/extra edges: " + SHDAdj);
	//		out.println("# different edge points: " + diffEdgePoint);
	//
	//		//		out.println("getEdgesAdded: " + comparison.getEdgesAdded());
	//		//		out.println("getEdgesRemoved: " + comparison.getEdgesRemoved());
	//		//		out.println("getEdgesReorientedFrom: " + comparison.getEdgesReorientedFrom());
	//		//		out.println("getEdgesReorientedTo: " + comparison.getEdgesReorientedTo());
	//		out.println("-------------------------------");
	//
	//		// return getTextTable(dataSet, cols, new DecimalFormat("0.00"));
	//		// //deleted .toString()
	//	}
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
			File dir = new File(RBExperiments_sim_old.directory + "/xdsl");
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

	//	private double[] printCorrectArrows(Graph outGraph, Graph truePag, PrintStream out) {
	//		int correctArrows = 0;
	//		int totalEstimatedArrows = 0;
	//		int totalTrueArrows = 0;
	//
	//		double[] stats = new double[5];
	//
	//		for (Edge edge : outGraph.getEdges()) {
	//			Node x = edge.getNode1();
	//			Node y = edge.getNode2();
	//
	//			Endpoint ex = edge.getEndpoint1();
	//			Endpoint ey = edge.getEndpoint2();
	//
	//			final Edge edge1 = truePag.getEdge(x, y);
	//
	//			if (ex == Endpoint.ARROW) {
	//				if (edge1 != null && edge1.getProximalEndpoint(x) == Endpoint.ARROW) {
	//					correctArrows++;
	//				}
	//				totalEstimatedArrows++;
	//			}
	//
	//			if (ey == Endpoint.ARROW) {
	//				if (edge1 != null && edge1.getProximalEndpoint(y) == Endpoint.ARROW) {
	//					correctArrows++;
	//				}
	//
	//				totalEstimatedArrows++;
	//			}
	//		}
	//
	//		for (Edge edge : truePag.getEdges()) {
	//			Endpoint ex = edge.getEndpoint1();
	//			Endpoint ey = edge.getEndpoint2();
	//
	//			if (ex == Endpoint.ARROW) {
	//				totalTrueArrows++;
	//			}
	//
	//			if (ey == Endpoint.ARROW) {
	//				totalTrueArrows++;
	//			}
	//		}
	//
	//		out.println();
	//		out.println("# correct arrows: " + correctArrows);
	//		out.println("# total estimated arrows: " + totalEstimatedArrows);
	//		out.println("# total true arrows: " + totalTrueArrows);
	//
	//		out.println();
	//		NumberFormat nf = new DecimalFormat("0.00");
	//		final double precision = correctArrows / (double) totalEstimatedArrows;
	//		out.println("Arrow precision: " + nf.format(precision));
	//		final double recall = correctArrows / (double) totalTrueArrows;
	//		out.println("Arrow recall: " + nf.format(recall));
	//
	//		stats[0] = correctArrows;
	//		stats[1] = totalEstimatedArrows;
	//		stats[2] = totalTrueArrows;
	//		stats[3] = precision;
	//		stats[4] = recall;
	//
	//		return stats;
	//	}

	//	private double[] printCorrectTails(Graph outGraph, Graph truePag, PrintStream out) {
	//		int correctTails = 0;
	//		int totalEstimatedTails = 0;
	//		int totalTrueTails = 0;
	//
	//		double[] stats = new double[5];
	//
	//		for (Edge edge : outGraph.getEdges()) {
	//			Node x = edge.getNode1();
	//			Node y = edge.getNode2();
	//
	//			Endpoint ex = edge.getEndpoint1();
	//			Endpoint ey = edge.getEndpoint2();
	//
	//			final Edge edge1 = truePag.getEdge(x, y);
	//
	//			if (ex == Endpoint.TAIL) {
	//				if (edge1 != null && edge1.getProximalEndpoint(x) == Endpoint.TAIL) {
	//					correctTails++;
	//				}
	//
	//				totalEstimatedTails++;
	//			}
	//
	//			if (ey == Endpoint.TAIL) {
	//				if (edge1 != null && edge1.getProximalEndpoint(y) == Endpoint.TAIL) {
	//					correctTails++;
	//				}
	//
	//				totalEstimatedTails++;
	//			}
	//		}
	//
	//		for (Edge edge : truePag.getEdges()) {
	//			Endpoint ex = edge.getEndpoint1();
	//			Endpoint ey = edge.getEndpoint2();
	//
	//			if (ex == Endpoint.TAIL) {
	//				totalTrueTails++;
	//			}
	//
	//			if (ey == Endpoint.TAIL) {
	//				totalTrueTails++;
	//			}
	//		}
	//
	//		out.println();
	//		out.println("# correct tails: " + correctTails);
	//		out.println("# total estimated tails: " + totalEstimatedTails);
	//		out.println("# total true tails: " + totalTrueTails);
	//
	//		out.println();
	//		NumberFormat nf = new DecimalFormat("0.00");
	//		final double precision = correctTails / (double) totalEstimatedTails;
	//		out.println("Tail precision: " + nf.format(precision));
	//		final double recall = correctTails / (double) totalTrueTails;
	//		out.println("Tail recall: " + nf.format(recall));
	//
	//		stats[0] = correctTails;
	//		stats[1] = totalEstimatedTails;
	//		stats[2] = totalTrueTails;
	//		stats[3] = precision;
	//		stats[4] = recall;
	//
	//		return stats;
	//	}

	//	private TextTable getTextTable(DataSet dataSet, int[] columns, NumberFormat nf) {
	//		TextTable table = new TextTable(dataSet.getNumRows() + 2, columns.length + 1);
	//
	//		table.setToken(0, 0, "Run #");
	//
	//		for (int j = 0; j < columns.length; j++) {
	//			table.setToken(0, j + 1, dataSet.getVariable(columns[j]).getName());
	//		}
	//
	//		for (int i = 0; i < dataSet.getNumRows(); i++) {
	//			table.setToken(i + 1, 0, Integer.toString(i + 1));
	//		}
	//
	//		for (int i = 0; i < dataSet.getNumRows(); i++) {
	//			for (int j = 0; j < columns.length; j++) {
	//				table.setToken(i + 1, j + 1, nf.format(dataSet.getDouble(i, columns[j])));
	//			}
	//		}
	//
	//		NumberFormat nf2 = new DecimalFormat("0.00");
	//
	//		for (int j = 0; j < columns.length; j++) {
	//			double sum = 0.0;
	//
	//			for (int i = 0; i < dataSet.getNumRows(); i++) {
	//				sum += dataSet.getDouble(i, columns[j]);
	//			}
	//
	//			double avg = sum / dataSet.getNumRows();
	//
	//			table.setToken(dataSet.getNumRows() + 2 - 1, j + 1, nf2.format(avg));
	//		}
	//
	//		table.setToken(dataSet.getNumRows() + 2 - 1, 0, "Avg");
	//
	//		return table;
	//	}

	private DataSet createDepDataFiltering(Map<IndependenceFact, Double> H, DataSet data, int numBootstrapSamples,
			boolean threshold, double lower, double upper) {
		List<Node> vars = new ArrayList<>();
		Map<IndependenceFact, Double> HCopy = new HashMap<>();
		for (IndependenceFact f : H.keySet()) {
			if (H.get(f) >= lower && H.get(f) <= upper) {
				HCopy.put(f, H.get(f));
				DiscreteVariable var = new DiscreteVariable(f.toString());
				vars.add(var);
			}
		}

		DataSet depData = new BoxDataSet(new DoubleDataBox(numBootstrapSamples, vars.size()), vars);
		System.out.println("\nDep data rows: " + depData.getNumRows() + ", columns: " + depData.getNumColumns());
		System.out.println("HCopy size: " + HCopy.size());

		for (int b = 0; b < numBootstrapSamples; b++) {
			DataSet bsData = DataUtils.getBootstrapSample(data, data.getNumRows());
			//			IndTestProbabilistic bsTest = new IndTestProbabilistic(bsData);
			IndTestProbabilisticBDeu bsTest = new IndTestProbabilisticBDeu(bsData, RBExperiments_sim_old.prior);
			bsTest.setThreshold(threshold);
			for (IndependenceFact f : HCopy.keySet()) {
				boolean ind = bsTest.isIndependent(f.getX(), f.getY(), f.getZ());
				int value = ind ? 1 : 0;
				depData.setInt(b, depData.getColumn(depData.getVariable(f.toString())), value);
			}
		}
		return depData;
	}

	private Graph runFGS(DataSet data) {
		BDeuScore sd = new BDeuScore(data);
		sd.setSamplePrior(1.0);
		sd.setStructurePrior(1.0);
		Fges fgs = new Fges(sd);
		fgs.setVerbose(false);
		fgs.setFaithfulnessAssumed(true);
		Graph fgsPattern = fgs.search();
		fgsPattern = GraphUtils.replaceNodes(fgsPattern, data.getVariables());
		return fgsPattern;
	}

	private allScores getLnProbsAll(List<Graph> pags, Map<IndependenceFact, Double> H, DataSet data, BayesIm im,
			Graph dep, Map<NodePair, GraphParameter> localBNs, PrintStream outlog) {
		// Map<Graph, Double> pagLnBDeu = new HashMap<Graph, Double>();
		Map<Graph, Double> pagLnBSCD = new HashMap<Graph, Double>();
		Map<Graph, Double> pagLnBSCLD = new HashMap<Graph, Double>();
		Map<Graph, Double> pagLnBSCI = new HashMap<Graph, Double>();

		for (int i = 0; i < pags.size(); i++) {
			Graph pagOrig = pags.get(i);
			if (!pagLnBSCD.containsKey(pagOrig)) {
				double lnInd = getLnProb(pagOrig, H);

				// Filtering
				double lnDep = getLnProbUsingDepFiltering(pagOrig, H, im, dep);
				double lnLocalDep = getLnProbUsingLocalDepFiltering(pagOrig, H, localBNs);

				pagLnBSCD.put(pagOrig, lnDep);
				pagLnBSCLD.put(pagOrig, lnLocalDep);
				pagLnBSCI.put(pagOrig, lnInd);
			}
		}
		outlog.println("pags size: " + pags.size());
		outlog.println("unique pags size: " + pagLnBSCD.size());

		System.out.println("pags size: " + pags.size());
		System.out.println("unique pags size: " + pagLnBSCD.size());

		return new allScores(pagLnBSCLD, pagLnBSCD, pagLnBSCI);
	}

	private class allScores {
		Map<Graph, Double> LnBSCLD;
		Map<Graph, Double> LnBSCD;
		Map<Graph, Double> LnBSCI;

		allScores(Map<Graph, Double> LnBSCLD, Map<Graph, Double> LnBSCD, Map<Graph, Double> LnBSCI) {
			this.LnBSCLD = LnBSCLD;
			this.LnBSCD = LnBSCD;
			this.LnBSCI = LnBSCI;
		}

	}

	private IndTestProbabilisticBDeu runRB(DataSet data, List<Graph> pags, int numModels, boolean threshold, Graph gs) {
		//		IndTestProbabilistic BSCtest = new IndTestProbabilistic(data);
		IndTestProbabilisticBDeu BSCtest = new IndTestProbabilisticBDeu(data, RBExperiments_sim_old.prior);
		BSCtest.setThreshold(threshold);
		//		BSCtest.setGoldStandard(gs);
		//		BSCtest.setIsMain(true);
		//		BicScore score = new BicScore(data);

		Fci BSCrfci = new Fci(BSCtest);

		BSCrfci.setVerbose(false);
		BSCrfci.setCompleteRuleSetUsed(RBExperiments_sim_old.completeRules);
		//		BSCrfci.setDepth(RBExperiments_sim.depth);

		for (int i = 0; i < numModels; i++) {
			if (i % 10 == 0)
				System.out.print(", i: " + i);
			Graph BSCPag = BSCrfci.search();
			BSCPag = GraphUtils.replaceNodes(BSCPag, data.getVariables());
			pags.add(BSCPag);

		}
		return BSCtest;
	}

	private Graph runPagCs(DataSet data, double alpha, Graph gs) {
		IndTestChiSquare test = new IndTestChiSquare(data, alpha);
		//		test.setGoldStandard(gs);
		//		test.setIsMain(true);
		//		BicScore score = new BicScore (data);

		Fci fci1 = new Fci(test);
		//		fci1.setDepth(RBExperiments_sim.depth);
		fci1.setVerbose(false);
		fci1.setCompleteRuleSetUsed(RBExperiments_sim_old.completeRules);
		Graph PAG_CS = fci1.search();
		PAG_CS = GraphUtils.replaceNodes(PAG_CS, data.getVariables());
		return PAG_CS;
	}

	// private double getLnProbUsingDep(Graph pag, Map<IndependenceFact, Double>
	// H, BayesIm im, Graph dep) {
	// double lnQ = 0;
	//
	// for (IndependenceFact fact : H.keySet()) {
	// BCInference.OP op;
	// double p = 0.0;
	//
	// if (pag.isDSeparatedFrom(fact.getX(), fact.getY(), fact.getZ())) {
	// op = BCInference.OP.independent;
	// } else {
	// op = BCInference.OP.dependent;
	// }
	//
	// Node node = im.getNode(fact.toString());
	//
	// int[] parents = im.getParents(im.getNodeIndex(node));
	//
	// if (parents.length > 0){
	//
	// int[] parentValues = new int[parents.length];
	//
	// for (int parentIndex = 0; parentIndex < parentValues.length;
	// parentIndex++) {
	// String parentName = im.getNode(parents[parentIndex]).getName();
	// String[] splitParent = parentName.split(Pattern.quote("_||_"));
	// Node X = pag.getNode(splitParent[0].trim());
	//
	// String[] splitParent2 = splitParent[1].trim().split(Pattern.quote("|"));
	// Node Y = pag.getNode(splitParent2[0].trim());
	//
	// List<Node> Z = new ArrayList<Node>();
	// if(splitParent2.length>1){
	// String[] splitParent3 = splitParent2[1].trim().split(Pattern.quote(","));
	// for(String s: splitParent3){
	// Z.add(pag.getNode(s.trim()));
	// }
	// }
	// IndependenceFact parentFact = new IndependenceFact(X, Y, Z);
	// if (pag.isDSeparatedFrom(parentFact.getX(), parentFact.getY(),
	// parentFact.getZ())) {
	// parentValues[parentIndex] = 1;
	// } else {
	// parentValues[parentIndex] = 0;
	// }
	// }
	//
	// int rowIndex = im.getRowIndex(im.getNodeIndex(node), parentValues);
	// p = im.getProbability(im.getNodeIndex(node), rowIndex, 1);
	//
	// if (op == BCInference.OP.dependent) {
	// p = 1.0 - p;
	// }
	// }
	// else{
	// p = im.getProbability(im.getNodeIndex(node), 0, 1);
	// if (op == BCInference.OP.dependent) {
	// p = 1.0 - p;
	// }
	// }
	//
	// if (p < -0.0001 || p > 1.0001 || Double.isNaN(p) || Double.isInfinite(p))
	// {
	// throw new IllegalArgumentException("p illegally equals " + p);
	// }
	//
	// double v = lnQ + log(p);
	//
	// if (Double.isNaN(v) || Double.isInfinite(v)) {
	// continue;
	// }
	//
	// lnQ = v;
	// }
	// return lnQ;
	// }

	private double getLnProbUsingDepFiltering(Graph pag, Map<IndependenceFact, Double> H, BayesIm im, Graph dep) {
		double lnQ = 0;

		for (IndependenceFact fact : H.keySet()) {
			BCInference.OP op;
			double p = 0.0;

			if (pag.isDSeparatedFrom(fact.getX(), fact.getY(), fact.getZ())) {
				op = BCInference.OP.independent;
			} else {
				op = BCInference.OP.dependent;
			}

			if (im.getNode(fact.toString()) != null) {
				Node node = im.getNode(fact.toString());

				int[] parents = im.getParents(im.getNodeIndex(node));

				if (parents.length > 0) {

					int[] parentValues = new int[parents.length];

					for (int parentIndex = 0; parentIndex < parentValues.length; parentIndex++) {
						String parentName = im.getNode(parents[parentIndex]).getName();
						String[] splitParent = parentName.split(Pattern.quote("_||_"));
						Node X = pag.getNode(splitParent[0].trim());

						String[] splitParent2 = splitParent[1].trim().split(Pattern.quote("|"));
						Node Y = pag.getNode(splitParent2[0].trim());

						List<Node> Z = new ArrayList<Node>();
						if (splitParent2.length > 1) {
							String[] splitParent3 = splitParent2[1].trim().split(Pattern.quote(","));
							for (String s : splitParent3) {
								Z.add(pag.getNode(s.trim()));
							}
						}
						IndependenceFact parentFact = new IndependenceFact(X, Y, Z);
						if (pag.isDSeparatedFrom(parentFact.getX(), parentFact.getY(), parentFact.getZ())) {
							parentValues[parentIndex] = 1;
						} else {
							parentValues[parentIndex] = 0;
						}
					}

					int rowIndex = im.getRowIndex(im.getNodeIndex(node), parentValues);
					p = im.getProbability(im.getNodeIndex(node), rowIndex, 1);

					if (op == BCInference.OP.dependent) {
						p = 1.0 - p;
					}
				} else {
					p = im.getProbability(im.getNodeIndex(node), 0, 1);
					if (op == BCInference.OP.dependent) {
						p = 1.0 - p;
					}
				}

				if (p < -0.0001 || p > 1.0001 || Double.isNaN(p) || Double.isInfinite(p)) {
					throw new IllegalArgumentException("p illegally equals " + p);
				}

				double v = lnQ + log(p);

				if (Double.isNaN(v) || Double.isInfinite(v)) {
					continue;
				}

				lnQ = v;
			} else {
				p = H.get(fact);

				if (p < -0.0001 || p > 1.0001 || Double.isNaN(p) || Double.isInfinite(p)) {
					throw new IllegalArgumentException("p illegally equals " + p);
				}

				if (op == BCInference.OP.dependent) {
					p = 1.0 - p;
				}

				double v = lnQ + log(p);

				if (Double.isNaN(v) || Double.isInfinite(v)) {
					continue;
				}

				lnQ = v;
			}
		}

		return lnQ;
	}
	private double getLnProbUsingLocalDepFiltering(Graph pag, Map<IndependenceFact, Double> H, 
			Map<NodePair, GraphParameter> localBNs) {
		double lnQ = 0;

		for (IndependenceFact fact : H.keySet()) {
			BCInference.OP op;
			double p = 0.0;

			if (pag.isDSeparatedFrom(fact.getX(), fact.getY(), fact.getZ())) {
				op = BCInference.OP.independent;
			} else {
				op = BCInference.OP.dependent;
			}

			//			NodePair ab = new NodePair(fact.getX().getName(), fact.getY().getName());

			//			BayesIm im = null;
			GraphParameter relevant_BN = null ;
			for (NodePair ab: localBNs.keySet()){
				if ( (fact.getX().getName().equals(ab.n_a) && fact.getY().getName().equals(ab.n_b)) || 
						(fact.getX().getName().equals(ab.n_b) && fact.getY().getName().equals(ab.n_a)) ){
					relevant_BN = localBNs.get(ab);
				}
			}
			BayesIm	im = null;
			if (relevant_BN != null){
				im = relevant_BN.theta;
			}

			if (im != null && im.getNode(fact.toString()) != null) {
				Node node = im.getNode(fact.toString());

				int[] parents = im.getParents(im.getNodeIndex(node));

				if (parents != null && parents.length > 0) {

					int[] parentValues = new int[parents.length];

					for (int parentIndex = 0; parentIndex < parentValues.length; parentIndex++) {
						String parentName = im.getNode(parents[parentIndex]).getName();
						String[] splitParent = parentName.split(Pattern.quote("_||_"));
						Node X = pag.getNode(splitParent[0].trim());

						String[] splitParent2 = splitParent[1].trim().split(Pattern.quote("|"));
						Node Y = pag.getNode(splitParent2[0].trim());

						List<Node> Z = new ArrayList<Node>();
						if (splitParent2.length > 1) {
							String[] splitParent3 = splitParent2[1].trim().split(Pattern.quote(","));
							for (String s : splitParent3) {
								Z.add(pag.getNode(s.trim()));
							}
						}
						IndependenceFact parentFact = new IndependenceFact(X, Y, Z);
						if (pag.isDSeparatedFrom(parentFact.getX(), parentFact.getY(), parentFact.getZ())) {
							parentValues[parentIndex] = 1;
						} else {
							parentValues[parentIndex] = 0;
						}
					}

					int rowIndex = im.getRowIndex(im.getNodeIndex(node), parentValues);
					p = im.getProbability(im.getNodeIndex(node), rowIndex, 1);

					if (op == BCInference.OP.dependent) {
						p = 1.0 - p;
					}
				} else {
					p = im.getProbability(im.getNodeIndex(node), 0, 1);
					if (op == BCInference.OP.dependent) {
						p = 1.0 - p;
					}
				}

				if (p < -0.0001 || p > 1.0001 || Double.isNaN(p) || Double.isInfinite(p)) {
					throw new IllegalArgumentException("p illegally equals " + p);
				}

				double v = lnQ + log(p);

				if (Double.isNaN(v) || Double.isInfinite(v)) {
					continue;
				}

				lnQ = v;
			} 
			else {
				p = H.get(fact);

				if (p < -0.0001 || p > 1.0001 || Double.isNaN(p) || Double.isInfinite(p)) {
					throw new IllegalArgumentException("p illegally equals " + p);
				}

				if (op == BCInference.OP.dependent) {
					p = 1.0 - p;
				}

				double v = lnQ + log(p);

				if (Double.isNaN(v) || Double.isInfinite(v)) {
					continue;
				}

				lnQ = v;
			}
		}

		return lnQ;
	}

	private double getLnProb(Graph pag, Map<IndependenceFact, Double> H) {
		double lnQ = 0;
		for (IndependenceFact fact : H.keySet()) {
			BCInference.OP op;

			if (pag.isDSeparatedFrom(fact.getX(), fact.getY(), fact.getZ())) {
				op = BCInference.OP.independent;
			} else {
				op = BCInference.OP.dependent;
			}

			double p = H.get(fact);

			if (p < -0.0001 || p > 1.0001 || Double.isNaN(p) || Double.isInfinite(p)) {
				throw new IllegalArgumentException("p illegally equals " + p);
			}

			if (op == BCInference.OP.dependent) {
				p = 1.0 - p;
			}

			double v = lnQ + log(p);

			if (Double.isNaN(v) || Double.isInfinite(v)) {
				continue;
			}

			lnQ = v;
		}
		return lnQ;
	}

	private Map<Graph, Double> normalProbs(Map<Graph, Double> pagLnProbs) {
		double lnQTotal = lnQTotal(pagLnProbs);
		Map<Graph, Double> normalized = new HashMap<Graph, Double>();
		for (Graph pag : pagLnProbs.keySet()) {
			double lnQ = pagLnProbs.get(pag);
			double normalizedlnQ = lnQ - lnQTotal;
			normalized.put(pag, Math.exp(normalizedlnQ));
		}
		return normalized;
	}

	// private Map<Graph, Double> getLnProbs(List<Graph> pags,
	// Map<IndependenceFact, Double> H) {
	// Map<Graph, Double> pagLnProb = new HashMap<Graph, Double>();
	// for (int i = 0; i < pags.size(); i++) {
	// Graph pag = pags.get(i);
	// double lnQ = getLnProb(pag, H);
	// pagLnProb.put(pag, lnQ);
	// }
	// System.out.println("pags size: " + pags.size());
	// System.out.println("unique pags size: " + pagLnProb.size());
	//
	// return pagLnProb;
	// }

	// private Map<Graph, Double> getLnProbsUsingDep(List<Graph> pags,
	// Map<IndependenceFact, Double> H, BayesIm imHat, Graph estDepBN) {
	// Map<Graph, Double> pagLnProb = new HashMap<Graph, Double>();
	// for (int i = 0; i < pags.size(); i++) {
	// Graph pag = pags.get(i);
	// double lnQ = getLnProbUsingDep(pag, H, imHat, estDepBN);
	// pagLnProb.put(pag, lnQ);
	// }
	// System.out.println("pags size: " + pags.size());
	// System.out.println("unique pags size: " + pagLnProb.size());
	//
	// return pagLnProb;
	// }

	protected double lnXplusY(double lnX, double lnY) {
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
			double w = Math.log1p(exp(lnYminusLnX));
			return w + lnX;
		}
	}

	private double lnQTotal(Map<Graph, Double> pagLnProb) {
		Set<Graph> pags = pagLnProb.keySet();
		Iterator<Graph> iter = pags.iterator();
		double lnQTotal = pagLnProb.get(iter.next());

		while (iter.hasNext()) {
			Graph pag = iter.next();
			double lnQ = pagLnProb.get(pag);
			lnQTotal = lnXplusY(lnQTotal, lnQ);
		}

		return lnQTotal;
	}

	private static final int MININUM_EXPONENT = -1022;

	public DataSet bootStrapSampling(DataSet data, int numBootstrapSamples, int bootsrapSampleSize) {

		DataSet bootstrapSample = DataUtils.getBootstrapSample(data, bootsrapSampleSize);
		return bootstrapSample;
	}

	private class NodePair{
		private final String n_a;
		private final String n_b;

		private NodePair(final String n_a, final String n_b) {
			this.n_a = n_a;
			this.n_b = n_b;
		}
		@Override
		public boolean equals (final Object O) {
			if (!(O instanceof NodePair)) return false;
			if (!((NodePair) O).n_a.equals(n_a)) return false;
			if (!((NodePair) O).n_b.equals(n_b)) return false;
			return true;
		}
		// @Override
		// public int hashCode() {
		//	 return this.n_a.getName() + this.n_d.getName();
		// }

		public String print(NodePair pair){
			return "("+pair.n_a +", "+ pair.n_b + ")";
		}

	}
	private class GraphParameter{
		private final Graph g;
		private final BayesIm theta;

		private GraphParameter(final Graph g, final BayesIm theta) {
			this.g = g;
			this.theta = theta;
		}
		@Override
		public boolean equals (final Object O) {
			if (!(O instanceof GraphParameter)) return false;
			if (((GraphParameter) O).g != g) return false;
			if (((GraphParameter) O).theta != theta) return false;
			return true;
		}
		//			public String print(NodePair pair){
		//			return "("+pair.n_a +", "+ pair.n_b + ")";
		//		}
	}
}



