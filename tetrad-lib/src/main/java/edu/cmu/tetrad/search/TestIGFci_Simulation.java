package edu.cmu.tetrad.search;


import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


import edu.cmu.tetrad.algcomparison.statistic.utils.AdjacencyConfusionIS;
import edu.cmu.tetrad.algcomparison.statistic.utils.ArrowConfusionIS;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.ISMlBayesIm;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TextTable;

public class TestIGFci_Simulation {
	private PrintStream out;
	private boolean completeRules = false;
	private boolean threshold = true;
	private double prior = 0.5;
	public static void main(String[] args) {
		 /***
		 Process and initialize the following input arguments that will be used in simulations:
		 - numVars: number of variables
		 - numCases: number of training cases (samples)
		 - numTests: number of test cases (samples)
		 - edgesPerNode: number of edges per node
		 - kappa: penalty term that penalizes edge differences between the population-wide and instance-specific graphs (Fattaneh's dissertation page 104, eq 4.17)
		   Note: we could define different penalty terms for added/deleted/reoriented edges, but I use the same penalty
		 - samplePrior: prior that is being used for the population-wide structure prior
		 - LV: fraction of latent variables
		 - numSim: number of simulations
		 ***/
		int numVars = 10, numCases = 500, numTests = 100, numSim = 5;
		double edgesPerNode = 2.0, kappa = 0.1, samplePrior = 1.0;
		String data_path = System.getProperty("user.dir") + "/IGFCI/";
		double LV = 0.2;
		for (int i = 0; i < args.length; i++) {
			switch (args[i]) {
			case "-c":
				numCases = Integer.parseInt(args[i + 1]);
				break;
			case "-t":
				numTests = Integer.parseInt(args[i + 1]);
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
			case "-kappa":
				kappa = Double.parseDouble(args[i + 1]);
				break;
			case "-pess":
				samplePrior = Double.parseDouble(args[i + 1]);
				break;
			case "-out":
				data_path = args[i + 1];
				break;
			case "-numSim":
				numSim = Integer.parseInt(args[i + 1]);
				break;
			}
		}
			// run simulation
			TestIGFci_Simulation t = new TestIGFci_Simulation();
			t.testSimulation(numVars,edgesPerNode, LV, numCases, numTests, kappa, samplePrior, numSim, data_path);

	}

	public void testSimulation(int numVars, double edgesPerNode, double LV, int numCases, 
			int numTests, double kappa, double samplePrior, int numSim, String data_path){
		// set the seed for reproducibility
		RandomUtil.getInstance().setSeed(1454147770L);

		// set the minimum and maximum number of categories
		int minCat = 2;
		int maxCat = 4;
		
		// calculate number of edges and latent variables
		final int numEdges = (int) (numVars * edgesPerNode);
		int numLatents = (int) Math.floor(numVars * LV);

		System.out.println("# nodes: " + numVars + ", # edges: "+ numEdges + ", # training: " + numCases + ", # test: "+ numTests);
		System.out.println("k add: " + kappa + ", delete: "+ kappa + ", reverse: " + kappa);
		
		// the following variables are used to record evaluation criteria
		double avgInDeg2 = 0.0, avgOutDeg2 = 0.0;

		double[] arrP = new double[numSim], arrR = new double[numSim], adjP = new double[numSim], adjR = new double[numSim], 
				added = new double[numSim], removed = new double[numSim], reoriented = new double[numSim],
				addedIS = new double[numSim], removedIS = new double[numSim], reorientedIS = new double[numSim], 
				addedOther = new double[numSim], removedOther = new double[numSim], reorientedOther = new double[numSim];

		double[] arrPI = new double[numSim], arrRI = new double[numSim], adjPI = new double[numSim], adjRI = new double[numSim], 
				addedI = new double[numSim], removedI = new double[numSim], reorientedI = new double[numSim],
				addedI_IS = new double[numSim], removedI_IS = new double[numSim], reorientedI_IS = new double[numSim], 
				addedI_Other = new double[numSim], removedI_Other = new double[numSim], reorientedI_Other = new double[numSim];

		double[] arrIP = new double[numSim], arrIR = new double[numSim], arrNP = new double[numSim], arrNR = new double[numSim];
		double[] arrIPI = new double[numSim], arrIRI = new double[numSim], arrNPI = new double[numSim], arrNRI = new double[numSim];

		double[] adjIP = new double[numSim], adjIR = new double[numSim], adjNP = new double[numSim], adjNR = new double[numSim];
		double[] adjIPI = new double[numSim], adjIRI = new double[numSim], adjNPI = new double[numSim], adjNRI = new double[numSim];
		
		
		double[] avgcsi = new double[numSim];
		double[] shdStrict = new double[numSim], shdLenient = new double[numSim], shdAdjacency = new double[numSim];
		double[] shdStrictI = new double[numSim], shdLenientI = new double[numSim], shdAdjacencyI = new double[numSim];
		
		// create the output directory to save the results
		try {
			File dir = new File(data_path + "/simulation-GFCI-kappa"+kappa+"/PESS"+samplePrior+"/");
			dir.mkdirs();
			String outputFileName = "V"+numVars +"-E"+ edgesPerNode + "-N"+ numCases + "-T"+ numTests + "-kappa"
					+ kappa+ "-PESS" + samplePrior + ".csv";
			File file = new File(dir, outputFileName);
			if (file.exists() && file.length() != 0){ 
				return;
			}else{
				this.out = new PrintStream(new FileOutputStream(file));
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

		// loop over the rounds of simulations
		for (int s = 0; s < numSim; s++){
			// set the seed for each simulation round	
			RandomUtil.getInstance().setSeed(1454147770L + 1000 * s);
				
			System.out.println("simulation: " + s);

			// these variables are used to keep track of undefined values of evaluation statistics
			double cArrIP = 0.0, cArrIR = 0.0, cArrNP = 0.0, cArrNR = 0.0, cArrP = 0.0, cArrR = 0.0;
			double cArrIPI = 0.0, cArrIRI = 0.0, cArrNPI = 0.0, cArrNRI = 0.0, cArrPI=0.0, cArrRI=0.0;
			double cAdjIP = 0.0, cAdjIR = 0.0, cAdjNP = 0.0, cAdjNR = 0.0, cAdjP = 0.0,  cAdjR = 0.0;
			double cAdjIPI = 0.0, cAdjIRI = 0.0, cAdjNPI = 0.0, cAdjNRI = 0.0,  cAdjPI = 0.0, cAdjRI = 0.0;

			// generate true BN and its parameters
			List<Node> vars = createVariables(numVars);
			Graph trueBN = GraphUtils.randomGraphRandomForwardEdges(vars, numLatents, numEdges, 10, 10, 10, false, true);
			BayesPm pm = new BayesPm(trueBN, minCat, maxCat);
			ISMlBayesIm im = new ISMlBayesIm(pm, ISMlBayesIm.RANDOM);

			// simulate train and test data from BN
			DataSet fullTrainData = im.simulateData(numCases, true);
			DataSet fullTestData = im.simulateData(numTests, true);

			// get the observed part of the data only
			DataSet trainData = DataUtils.restrictToMeasured(fullTrainData);
			DataSet testData = DataUtils.restrictToMeasured(fullTestData);

			// learn the population-wide model
			BDeuScore scoreP = new BDeuScore(trainData);
			scoreP.setSamplePrior(samplePrior);
			IndTestProbabilisticBDeu2 BSCtest = new IndTestProbabilisticBDeu2(trainData, this.prior );
			BSCtest.setThreshold(this.threshold);
			GFci fgesP = new GFci (BSCtest, scoreP);
			Graph graphP = fgesP.search();
			graphP = GraphUtils.replaceNodes(graphP, trainData.getVariables());
			
			// learn the instance-specific model for each sample in the test set
			for (int i = 0; i < testData.getNumRows(); i++){
				// obtain the i'th sample from the full and partial test datasets
				DataSet test = testData.subsetRows(new int[]{i});
				DataSet fullTest = fullTestData.subsetRows(new int[]{i});

				// obtain the ground-truth instance-specific BN based on CSI structures in the current test sample
				Map <Node, Boolean> context= new HashMap<Node, Boolean>();
				Graph trueBNI = SearchGraphUtils.patternForDag(new EdgeListGraph(GraphUtils.getISGraph(trueBN, im, fullTest, context)));
				
				// get the ground-truth underlying PAG of the ground-truth instance-specific BN
				final DagToPag2 dagToPag = new DagToPag2(trueBNI);
				dagToPag.setCompleteRuleSetUsed(this.completeRules);
				Graph PAG_True = dagToPag.convert();
				PAG_True = GraphUtils.replaceNodes(PAG_True, trueBNI.getNodes());
				
				// this is to compute the average percentage of CSI structures
				for (Node n: context.keySet()){
					if (context.get(n)){
						avgcsi[s] += 1;
					}
				}

				// define the instance-specific score 
				ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
				scoreI.setKAddition(kappa);
				scoreI.setKDeletion(kappa);
				scoreI.setKReorientation(kappa);
				scoreI.setSamplePrior(samplePrior);
				
				// define the instance-specific BSC test
				IndTestProbabilisticISBDeu2 testI = new IndTestProbabilisticISBDeu2(trainData, test, BSCtest.getH(), graphP);
				
				// run the IGFci algorithm 
				IGFci fgesI = new IGFci(testI, scoreI, fgesP.FgesGraph);
				Graph graphI = fgesI.search();
				graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());

				// evaluation of the instance-specific model
				ArrowConfusionIS congI = new ArrowConfusionIS(PAG_True, GraphUtils.replaceNodes(graphI, PAG_True.getNodes()), context);
				AdjacencyConfusionIS conAdjGI = new AdjacencyConfusionIS(PAG_True, GraphUtils.replaceNodes(graphI, PAG_True.getNodes()), context);

				// Arr - for nodes with CSI
				double denP = (congI.getArrowsITp() + congI.getArrowsIFp());
				double denR = (congI.getArrowsITp() + congI.getArrowsIFn());

				if (denP != 0.0){
					arrIPI[s] += (congI.getArrowsITp() / denP);
				}
				else
				{ 
					cArrIPI+= 1.0;	
				}
				
				if (denR != 0.0){
					arrIRI[s] += (congI.getArrowsITp() / denR);
				}
				else
				{ 
					cArrIRI+= 1.0;
				}

				// Arr - for nodes withOUT CSI
				denP = (congI.getArrowsNTp() + congI.getArrowsNFp());
				denR = (congI.getArrowsNTp() + congI.getArrowsNFn());
				if (denP != 0.0){
					arrNPI[s] += (congI.getArrowsNTp() / denP);
				}
				else
				{ 
					cArrNPI+= 1.0;
				}
				if (denR != 0.0){
					arrNRI[s] += (congI.getArrowsNTp() / denR);
				}
				else
				{ 
					cArrNRI+= 1.0;
				}

				// Arr over all nodes 
				denP = (congI.getArrowsTp()+congI.getArrowsFp());
				denR = (congI.getArrowsTp()+congI.getArrowsFn());

				if (denP != 0.0){
					arrPI[s] += (congI.getArrowsTp() / denP);
				}
				else
				{ 
					cArrPI+= 1.0;
				}
				if (denR != 0.0){
					arrRI[s] += (congI.getArrowsTp() / denR);
				}
				else
				{ 
					cArrRI+= 1.0;
				}

				// Adj for nodes with CSI
				denP = (conAdjGI.getAdjITp() + conAdjGI.getAdjIFp());
				denR = (conAdjGI.getAdjITp() + conAdjGI.getAdjIFn());

				if (denP != 0.0){
					adjIPI[s] += (conAdjGI.getAdjITp() / denP);
				}
				else
				{ 
					cAdjIPI+= 1.0;
				}
				if(denR != 0.0){
					adjIRI[s] += (conAdjGI.getAdjITp() / denR);
				}
				else
				{ 
					cAdjIRI+= 1.0;
				}

				// Adj for nodes withOUT CSI
				denP = (conAdjGI.getAdjNTp() + conAdjGI.getAdjNFp());
				denR = (conAdjGI.getAdjNTp() + conAdjGI.getAdjNFn());

				if (denP != 0.0){
					adjNPI[s] += (conAdjGI.getAdjNTp() / denP);
				}
				else
				{ 
					cAdjNPI+= 1.0;
				}
				if (denR != 0.0){
					adjNRI[s] += (conAdjGI.getAdjNTp() / denR);
				}
				else
				{ 
					cAdjNRI+= 1.0;
				}

				// Adj over all nodes 
				denP = (conAdjGI.getAdjTp() + conAdjGI.getAdjFp());
				denR = (conAdjGI.getAdjTp() + conAdjGI.getAdjFn());

				if (denP != 0.0){
					adjPI[s] += (conAdjGI.getAdjTp() / denP);
				}
				else
				{ 
					cAdjPI+= 1.0;
				}
				if (denR != 0.0){
					adjRI[s] += (conAdjGI.getAdjTp() / denR);
				}
				else
				{ 
					cAdjRI+= 1.0;
				}


				// evaluation of population-wide model 
				ArrowConfusionIS cong = new ArrowConfusionIS(PAG_True, GraphUtils.replaceNodes(graphP, PAG_True.getNodes()), context);
				AdjacencyConfusionIS conAdjG = new AdjacencyConfusionIS(PAG_True, GraphUtils.replaceNodes(graphP, PAG_True.getNodes()), context);

				// Arr for nodes with CSI
				denP = (cong.getArrowsITp() + cong.getArrowsIFp());
				denR = (cong.getArrowsITp() + cong.getArrowsIFn());

				if (denP != 0.0){
					arrIP[s] += (cong.getArrowsITp() / denP);
				}
				else
				{ 
					cArrIP+= 1.0;
				}
				if(denR != 0.0){
					arrIR[s] += (cong.getArrowsITp() / denR);
				}
				else
				{ 
					cArrIR+= 1.0;
				}

				// Arr for nodes withOUT CSI
				denP = (cong.getArrowsNTp() + cong.getArrowsNFp());
				denR = (cong.getArrowsNTp() + cong.getArrowsNFn());

				if (denP != 0.0){
					arrNP[s] += (cong.getArrowsNTp() / denP);
				}
				else
				{ 
					cArrNP+= 1.0;
				}
				if (denR != 0.0){
					arrNR[s] += (cong.getArrowsNTp() / denR);
				}
				else
				{ 
					cArrNR+= 1.0;
				}

				// Arr - over all nodes 
				denP = (cong.getArrowsTp() + cong.getArrowsFp());
				denR = (cong.getArrowsTp() + cong.getArrowsFn());

				if (denP != 0.0){
					arrP[s] += (cong.getArrowsTp() / denP);
				}
				else
				{ 
					cArrP+= 1.0;
				}
				if (denR != 0.0){
					arrR[s] += (cong.getArrowsTp() / denR);
				}
				else
				{ 
					cArrR+= 1.0;
				}

				// Adj for nodes with CSI
				denP = (conAdjG.getAdjITp() + conAdjG.getAdjIFp());
				denR = (conAdjG.getAdjITp() + conAdjG.getAdjIFn());

				if (denP != 0.0){
					adjIP[s] += (conAdjG.getAdjITp() / denP);
				}
				else
				{ 
					cAdjIP+= 1.0;
				}
				if(denR != 0.0){
					adjIR[s] += (conAdjG.getAdjITp() / denR);
				}
				else
				{ 
					cAdjIR+= 1.0;
				}

				// Adj for nodes withOUT CSI
				denP = (conAdjG.getAdjNTp() + conAdjG.getAdjNFp());
				denR = (conAdjG.getAdjNTp() + conAdjG.getAdjNFn());

				if (denP != 0.0){
					adjNP[s] += (conAdjG.getAdjNTp() / denP);
				}
				else
				{ 
					cAdjNP+= 1.0;
				}
				if (denR != 0.0){
					adjNR[s] += (conAdjG.getAdjNTp() / denR);
				}
				else
				{ 
					cAdjNR+= 1.0;
				}

				// Adj - over all nodes 
				denP = (conAdjG.getAdjTp() + conAdjG.getAdjFp());
				denR = (conAdjG.getAdjTp() + conAdjG.getAdjFn());

				if (denP != 0.0){
					adjP[s] += (conAdjG.getAdjTp() / denP);
				}
				else
				{ 
					cAdjP+= 1.0;
				}

				if (denR != 0.0){
					adjR[s] += (conAdjG.getAdjTp() / denR);
				}
				else
				{ 
					cAdjR+= 1.0;
				}


				// compute structural Hamming distance
				GraphUtils.GraphComparison cmpI = SearchGraphUtils.getGraphComparison(graphI, PAG_True, true);
				GraphUtils.GraphComparison cmpP = SearchGraphUtils.getGraphComparison(graphP, PAG_True, true);

				GraphUtils.GraphComparison cmpI2 = SearchGraphUtils.getGraphComparison(graphI, PAG_True, context);
				GraphUtils.GraphComparison cmpP2 = SearchGraphUtils.getGraphComparison(graphP, PAG_True, context);

				addedI_IS[s] += cmpI2.getEdgesAddedIS().size();
				removedI_IS[s] += cmpI2.getEdgesRemovedIS().size();
				reorientedI_IS[s] += cmpI2.getEdgesReorientedToIS().size();
				addedI_Other[s] += cmpI2.getEdgesAddedOther().size();
				removedI_Other[s] += cmpI2.getEdgesRemovedOther().size();
				reorientedI_Other[s] += cmpI2.getEdgesReorientedToOther().size();
				addedI[s] += cmpI2.getEdgesAdded().size();
				removedI[s] += cmpI2.getEdgesRemoved().size();
				reorientedI[s] += cmpI2.getEdgesReorientedTo().size();

				shdStrictI[s] += cmpI.getShdStrict();
				shdLenientI[s] += cmpI.getShdLenient();
				shdAdjacencyI[s] += cmpI2.getEdgesAdded().size() + cmpI2.getEdgesRemoved().size();

				addedIS[s] += cmpP2.getEdgesAddedIS().size();
				removedIS[s] += cmpP2.getEdgesRemovedIS().size();
				reorientedIS[s] += cmpP2.getEdgesReorientedToIS().size();
				addedOther[s] += cmpP2.getEdgesAddedOther().size();
				removedOther[s] += cmpP2.getEdgesRemovedOther().size();
				reorientedOther[s] += cmpP2.getEdgesReorientedToOther().size();
				added[s] += cmpP2.getEdgesAdded().size();
				removed[s] += cmpP2.getEdgesRemoved().size();
				reoriented[s] += cmpP2.getEdgesReorientedTo().size();

				shdStrict[s] += cmpP.getShdStrict();
				shdLenient[s] += cmpP.getShdLenient();
				shdAdjacency[s] += cmpP2.getEdgesAdded().size() + cmpP2.getEdgesRemoved().size();

			}
			
			avgcsi[s] /= (numVars * numTests);
			System.out.println("avg CSI : "+ avgcsi[s]);
			
			arrIPI[s] /= (numTests - cArrIPI);
			arrIRI[s] /= (numTests - cArrIRI);
			arrNPI[s] /= (numTests - cArrNPI);
			arrNRI[s] /= (numTests - cArrNRI);
			arrPI[s] /= (numTests - cArrPI);
			arrRI[s] /= (numTests - cArrRI);

			adjIPI[s] /= (numTests - cAdjIPI);
			adjIRI[s] /= (numTests - cAdjIRI);
			adjNPI[s] /= (numTests - cAdjNPI);
			adjNRI[s] /= (numTests - cAdjNRI);
			adjPI[s] /= (numTests - cAdjPI);
			adjRI[s] /= (numTests - cAdjRI);

			addedI[s] /= numTests;
			removedI[s] /= numTests;
			reorientedI[s] /= numTests;
			shdStrictI[s] /= numTests;
			shdLenientI[s] /= numTests;
			shdAdjacencyI[s] /= numTests;

			addedI_IS[s] /= numTests;
			removedI_IS[s] /= numTests;
			reorientedI_IS[s] /= numTests;
			addedI_Other[s] /= numTests;
			removedI_Other[s] /= numTests;
			reorientedI_Other[s] /= numTests;

			arrIP[s] /= (numTests - cArrIP);
			arrIR[s] /= (numTests - cArrIR);
			arrNP[s] /= (numTests - cArrNP);
			arrNR[s] /= (numTests - cArrNR);
			arrP[s] /= (numTests - cArrP);
			arrR[s] /= (numTests - cArrR);

			adjIP[s] /= (numTests - cAdjIP);
			adjIR[s] /= (numTests - cAdjIR);
			adjNP[s] /= (numTests - cAdjNP);
			adjNR[s] /= (numTests - cAdjNR);
			adjP[s] /= (numTests - cAdjP);
			adjR[s] /= (numTests - cAdjR);

			added[s] /= numTests;
			removed[s] /= numTests;
			reoriented[s] /= numTests;
			shdStrict[s] /= numTests;
			shdLenient[s] /= numTests;
			shdAdjacency[s] /= numTests;

			addedIS[s] /= numTests;
			removedIS[s] /= numTests;
			reorientedIS[s] /= numTests;
			addedOther[s] /= numTests;
			removedOther[s] /= numTests;
			reorientedOther[s] /= numTests;


		}
		// write the results in output file
		printRes(this.out, "CSI", numSim, arrIPI, arrNPI, arrPI, arrIRI, arrNRI, arrRI, adjIPI, adjNPI, adjPI,
				adjIRI, adjNRI, adjRI, addedI, removedI, reorientedI, addedI_IS, removedI_IS, reorientedI_IS, 
				addedI_Other, removedI_Other, reorientedI_Other, shdStrictI, shdLenientI, shdAdjacencyI);

		printRes(this.out,"POP", numSim, arrIP, arrNP, arrP, arrIR, arrNR, arrR, adjIP, adjNP, adjP, adjIR, adjNR,
				adjR, added, removed, reoriented, addedIS, removedIS, reorientedIS, addedOther, removedOther, 
				reorientedOther, shdStrict, shdLenient, shdAdjacency);
		this.out.close();

		System.out.println("----------------------");

	}
	// a function that creates variables
	private List<Node> createVariables(int numVars) {
		List<Node> vars = new ArrayList<>();
		for (int i = 0; i < numVars; i++) {
			vars.add(new DiscreteVariable("X" + i));
		}
		return vars;
	}
	
	// a function that writes the results in the output file
	private void printRes(PrintStream out, String alg, int numSim, double[] arrIPI, double[] arrNPI, 
			double[] arrPI, double[] arrIRI, double[] arrNRI, double[] arrRI, double[] adjIPI, 
			double[] adjNPI, double[] adjPI, double[] adjIRI, double[] adjNRI, double[] adjRI, 
			double[] addedI, double[] removedI, double[] reorientedI, double[] addedI_IS, double[] 
					removedI_IS, double[] reorientedI_IS, double[] addedI_Other, double[] removedI_Other,
					double[] reorientedI_Other, double[] shdStrictI, double[] shdLenientI, double[]shdAdjacencyI){
		NumberFormat nf = new DecimalFormat("0.00");

		TextTable table = new TextTable(numSim+2, 8);
		table.setTabDelimited(true);
		String header = ", adj_P_IS, adj_P_NS, adj_P, adj_R_IS, adj_R_NS, adj_R, arr_P_IS, arr_P_NS, arr_P,"
				+ " arr_R_IS, arr_R_NS, arr_R, added_IS, added_NS, added, removed_IS, removed_NS, removed, "
				+ "reoriented_IS, reoriented_NS, reoriented, S-SHD, L-SHD, A-SHD";
		table.setToken(0, 0, alg);
		table.setToken(0, 1, header);
		double arrIP = 0.0, arrNP = 0.0, arrP = 0.0, arrIR = 0.0, arrNR = 0.0, arrR = 0.0,
				adjIP = 0.0, adjNP = 0.0, adjP = 0.0, adjIR = 0.0, adjNR = 0.0, adjR = 0.0,
				added = 0.0, removed = 0.0, reoriented = 0.0,
				addedIS = 0.0, removedIS = 0.0, reorientedIS = 0.0,
				addedNS = 0.0, removedNS = 0.0, reorientedNS = 0.0, sshd = 0.0, lshd = 0.0, ashd = 0.0;
		for (int i = 0; i < numSim; i++){
			String res = "," +nf.format(adjIPI[i])+","+nf.format(adjNPI[i])+","+nf.format(adjPI[i])+","+ nf.format(adjIRI[i])+
					","+nf.format(adjNRI[i])+","+nf.format(adjRI[i])+","+
					nf.format(arrIPI[i])+","+nf.format(arrNPI[i])+","+nf.format(arrPI[i])+","+ nf.format(arrIRI[i])+
					","+nf.format(arrNRI[i])+","+nf.format(arrRI[i])+","+
					nf.format(addedI_IS[i])+","+nf.format(addedI_Other[i])+","+nf.format(addedI[i])+","+
					nf.format(removedI_IS[i])+","+nf.format(removedI_Other[i])+","+nf.format(removedI[i])+","+
					nf.format(reorientedI_IS[i])+","+nf.format(reorientedI_Other[i])+","+nf.format(reorientedI[i])+","+
					nf.format(shdStrictI[i])+","+nf.format(shdLenientI[i])+","+nf.format(shdAdjacencyI[i]);
			table.setToken(i+1, 0, ""+(i+1));
			table.setToken(i+1, 1, res);
			arrIP += arrIPI[i];
			arrNP += arrNPI[i];
			arrP += arrPI[i];
			arrIR += arrIRI[i];
			arrNR += arrNRI[i];
			arrR += arrRI[i];
			adjIP += adjIPI[i];
			adjNP += adjNPI[i];
			adjP += adjPI[i];
			adjIR += adjIRI[i];
			adjNR += adjNRI[i];
			adjR += adjRI[i];
			added += addedI[i];
			removed += removedI[i];
			reoriented += reorientedI[i];
			addedIS += addedI_IS[i];
			removedIS += removedI_IS[i];
			reorientedIS += reorientedI_IS[i];
			addedNS += addedI_Other[i];
			removedNS += removedI_Other[i];
			reorientedNS += reorientedI_Other[i];
			sshd += shdStrictI[i];
			lshd += shdLenientI[i];
			ashd += shdAdjacencyI[i];
		}
		String res =  ","+nf.format(adjIP/numSim)+","+nf.format(adjNP/numSim)+","+nf.format(adjP/numSim)+","+nf.format(adjIR/numSim)+","+nf.format(adjNR/numSim)+","+nf.format(adjR/numSim)+","+
				nf.format(arrIP/numSim)+","+nf.format(arrNP/numSim)+","+nf.format(arrP/numSim)+","+nf.format(arrIR/numSim)+","+nf.format(arrNR/numSim)+","+nf.format(arrR/numSim)+","+
				nf.format(addedIS/numSim)+","+nf.format(addedNS/numSim)+","+nf.format(added/numSim)+","+
				nf.format(removedIS/numSim)+","+nf.format(removedNS/numSim)+","+nf.format(removed/numSim)+","+
				nf.format(reorientedIS/numSim)+","+nf.format(reorientedNS/numSim)+","+nf.format(reoriented/numSim)+","+
				nf.format(sshd/numSim)+","+nf.format(lshd/numSim)+","+nf.format(ashd/numSim);
		table.setToken(numSim+1, 0, "avg");
		table.setToken(numSim+1, 1, res);
		out.println(table);
		System.out.println(table);		
	}

}
