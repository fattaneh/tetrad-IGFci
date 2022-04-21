package edu.cmu.tetrad.test;


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
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TextTable;
//import nu.xom.Builder;
//import nu.xom.Document;
//import nu.xom.ParsingException;

public class TestISFges_Simulation {
	private PrintStream out;
	public static void main(String[] args) {
		int[] numVarss = new int[]{20};
		double[] edgesPerNodes = new double[]{4.0};
		int[] numCasess = {5000};
		for (int numVars: numVarss){
			for (double edgesPerNode : edgesPerNodes){
				for (int numCases : numCasess){
					TestISFges_Simulation t = new TestISFges_Simulation();
					t.testSimulation(numVars,edgesPerNode, numCases);
				}
			}
		}
	}

	public void testSimulation(int numVars, double edgesPerNode, int numCases){

		int numTests = 500;
		int minCat = 2;
		int maxCat = 4;
		int numSim = 1;
		double k_add = 0.9;
		double k_delete = k_add; 
		double k_reverse = k_add; 
		double samplePrior = 1.0;
		
		final int numEdges = (int) (numVars * edgesPerNode);
		double avgInDeg2 = 0.0, avgOutDeg2 = 0.0;
		System.out.println("# nodes: " + numVars + ", # edges: "+ numEdges + ", # training: " + numCases + ", # test: "+ numTests);
		System.out.println("k add: " + k_add + ", delete: "+ k_delete + ", reverse: " + k_reverse);
		double[] arrP = new double[numSim], arrR = new double[numSim], adjP = new double[numSim], adjR = new double[numSim], 
				added = new double[numSim], removed = new double[numSim], reoriented = new double[numSim],
				addedIS = new double[numSim], removedIS = new double[numSim], reorientedIS = new double[numSim], 
				addedOther = new double[numSim], removedOther = new double[numSim], reorientedOther = new double[numSim], llr = new double[numSim];

		double[] arrPI = new double[numSim], arrRI = new double[numSim], adjPI = new double[numSim], adjRI = new double[numSim], 
				addedI = new double[numSim], removedI = new double[numSim], reorientedI = new double[numSim],
				addedI_IS = new double[numSim], removedI_IS = new double[numSim], reorientedI_IS = new double[numSim], 
				addedI_Other = new double[numSim], removedI_Other = new double[numSim], reorientedI_Other = new double[numSim];

		double[] arrIP = new double[numSim], arrIR = new double[numSim], arrNP = new double[numSim], arrNR = new double[numSim];
		double[] arrIPI = new double[numSim], arrIRI = new double[numSim], arrNPI = new double[numSim], arrNRI = new double[numSim];

		double[] adjIP = new double[numSim], adjIR = new double[numSim], adjNP = new double[numSim], adjNR = new double[numSim];
		double[] adjIPI = new double[numSim], adjIRI = new double[numSim], adjNPI = new double[numSim], adjNRI = new double[numSim];
		double[] avgcsi = new double[numSim];
		try {
			File dir = new File("/Users/fattanehjabbari/CCD-Project/CS-BN/dissertation/simulation-IGES-kappa"+k_add+"/PESS"+samplePrior+"/");

			dir.mkdirs();
			String outputFileName = "V"+numVars +"-E"+ edgesPerNode + "-N"+ numCases + "-T"+ numTests + "-kappa" + k_add+ "-PESS" + samplePrior+"-np.csv";
			File file = new File(dir, outputFileName);
			if (file.exists() && file.length() != 0){ 
				return;
			}else{
				this.out = new PrintStream(new FileOutputStream(file));
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

		// loop over simulations
		for (int s = 0; s < numSim; s++){

			RandomUtil.getInstance().setSeed(1454147770L + 1000 * s);

			System.out.println("simulation: " + s);

			// create variables
			IKnowledge knowledge = new Knowledge2();
			List<Node> vars = new ArrayList<>();
			int[] tiers = new int[numVars];
			for (int i = 0; i < numVars; i++) {
				vars.add(new DiscreteVariable("X" + i));
				tiers[i] = i;
				knowledge.addToTier(i, "X" + i);
			}

			// generate true BN and its parameters
			Graph trueBN = GraphUtils.randomGraphRandomForwardEdges(vars, 0, numEdges, 10, 10, 10, false, true);
			//for (Node nod: trueBN.getNodes()){
			//	if (trueBN.getIndegree(nod)>1)
			//		avgInDeg2 = avgInDeg2 + 1;
			//	if (trueBN.getOutdegree(nod)>1)
			//		avgOutDeg2  = avgOutDeg2 +1;
			//}
			BayesPm pm = new BayesPm(trueBN, minCat, maxCat);
			ISMlBayesIm im = new ISMlBayesIm(pm, ISMlBayesIm.RANDOM);
			System.out.println(trueBN);
//			this.out.println("im: " + im);
//			System.out.println(SearchGraphUtils.patternFromDag(trueBN));

			
			// simulate train and test data from BN
			DataSet trainData = im.simulateData(numCases, false, tiers);
			DataSet testData = im.simulateData(numTests, false, tiers);

			// learn the population model
			BDeuScore scoreP = new BDeuScore(trainData);
			scoreP.setSamplePrior(samplePrior);
			Fges fgesP = new Fges (scoreP);
			fgesP.setVerbose(false);
			//fgesP.setKnowledge(knowledge);
			Graph graphP = fgesP.search();

			// estimate MAP parameters from the population model
			DagInPatternIterator iterator = new DagInPatternIterator(graphP);
			Graph dagP = iterator.next();
			//dagP = GraphUtils.replaceNodes(dagP, trainData.getVariables());

			//BayesPm pmP = new BayesPm(dagP);
			////			BayesPm pmP = new BayesPm(graphP);
			//DirichletBayesIm priorP = DirichletBayesIm.symmetricDirichletIm(pmP, 1.0);
			//BayesIm imP = DirichletEstimator.estimate(priorP, trainData);
			//			System.out.println("trueBN: " + trueBN);
//			double arrIRc = 0.0, arrNRc = 0.0, arrRc = 0.0, arrIRIc = 0.0, arrNRIc = 0.0, arrRIc = 0.0;
//			double adjIRc = 0.0, adjNRc = 0.0, adjRc = 0.0, adjIRIc = 0.0, adjNRIc = 0.0, adjRIc = 0.0;
			double csi = 0.0;
			for (int i = 0; i < testData.getNumRows(); i++){
				DataSet test = testData.subsetRows(new int[]{i});
				if (i%100 == 0) {System.out.println(i + " test instances done!");}
					System.out.println("test: " + test);

				// obtain the true instance-specific BN
				Map <Node, Boolean> context= new HashMap<Node, Boolean>();
				Graph trueBNI = SearchGraphUtils.patternForDag(new EdgeListGraph(GraphUtils.getISGraph(trueBN, im, test, context)));
				System.out.println("context: " + context);
				System.out.println("trueBNI: " + trueBNI);

				for (Node n: context.keySet()){
					if (context.get(n)){
						avgcsi[s] += 1;
					}
				}

				// learn the instance-specific model
				ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
				scoreI.setKAddition(k_add);
				scoreI.setKDeletion(k_delete);
				scoreI.setKReorientation(k_reverse);
				scoreI.setSamplePrior(samplePrior);
				ISFges fgesI = new ISFges(scoreI);
				fgesI.setPopulationGraph(graphP);
				fgesI.setInitialGraph(graphP);
				Graph graphI = fgesI.search();

				ArrowConfusionIS congI = new ArrowConfusionIS(trueBNI, GraphUtils.replaceNodes(graphI, trueBNI.getNodes()), context);
				AdjacencyConfusionIS conAdjGI = new AdjacencyConfusionIS(trueBNI, GraphUtils.replaceNodes(graphI, trueBNI.getNodes()), context);

				// arr - nodes w/ CSI
				double denP = (congI.getArrowsITp() + congI.getArrowsIFp());
				double denR = (congI.getArrowsITp() + congI.getArrowsIFn());
				if (denP == 0.0 && denR == 0.0){
					arrIPI[s] += 1.0;
					arrIRI[s] += 1.0;
				}
				if (denP != 0.0){
					arrIPI[s] += (congI.getArrowsITp() / denP);
				}
				if (denR != 0.0){
					arrIRI[s] += (congI.getArrowsITp() / denR);
				}

				// arr - nodes w/o CSI	
				denP = (congI.getArrowsNTp() + congI.getArrowsNFp());
				denR = (congI.getArrowsNTp() + congI.getArrowsNFn());
				if (denP == 0.0 && denR == 0.0){
					arrNPI[s] += 1.0;
					arrNRI[s] += 1.0;
//					arrNPIc += 1;
//					arrNRI += 1;
				}
				if (denP != 0.0){
//					arrNPIc += 1;
					arrNPI[s] += (congI.getArrowsNTp() / denP);
				}
				if (denR != 0.0){
//					arrNRI += 1;
					arrNRI[s] += (congI.getArrowsNTp() / denR);
				}
				
				// arr - over all nodes 
				denP = (congI.getArrowsTp()+congI.getArrowsFp());
				denR = (congI.getArrowsTp()+congI.getArrowsFn());
				if (denP == 0.0 && denR == 0.0){
					arrPI[s] += 1.0;
					arrRI[s] += 1.0;
//					arrPIc += 1;
//					arrRIc += 1;
				}
				if (denP != 0.0){
					arrPI[s] += (congI.getArrowsTp() / denP);
//					arrPIc += 1;
				}
				if (denR != 0.0){
					arrRI[s] += (congI.getArrowsTp() / denR);
//					arrRIc += 1;
				}

				// adj - nodes w/ CSI
				denP = (conAdjGI.getAdjITp() + conAdjGI.getAdjIFp());
				denR = (conAdjGI.getAdjITp() + conAdjGI.getAdjIFn());
				if (denP == 0.0 && denR == 0.0){
					adjIPI[s] += 1.0;
					adjIRI[s] += 1.0;
//					adjIPIc += 1;
//					adjIRIc += 1;
				}
				if (denP != 0.0){
					adjIPI[s] += (conAdjGI.getAdjITp() / denP);
//					adjIPIc += 1;
				}
				if(denR != 0.0){
					adjIRI[s] += (conAdjGI.getAdjITp() / denR);
//					adjIRIc += 1;
				}
				
				// adj - nodes w/o CSI
				denP = (conAdjGI.getAdjNTp() + conAdjGI.getAdjNFp());
				denR = (conAdjGI.getAdjNTp() + conAdjGI.getAdjNFn());
				if (denP == 0.0 && denR == 0.0){
					adjNPI[s] += 1.0;
					adjNRI[s] += 1.0;
//					adjNPIc += 1;
//					adjNRIc += 1;
				}
				if (denP != 0.0){
					adjNPI[s] += (conAdjGI.getAdjNTp() / denP);
//					adjNPIc += 1;
				}
				if (denR != 0.0){
					adjNRI[s] += (conAdjGI.getAdjNTp() / denR);
//					adjNRIc += 1;
				}

				// adj - over all nodes 
				denP = (conAdjGI.getAdjTp() + conAdjGI.getAdjFp());
				denR = (conAdjGI.getAdjTp() + conAdjGI.getAdjFn());
				if (denP == 0.0 && denR == 0.0){
					adjPI[s] += 1.0;
					adjRI[s] += 1.0;
//					adjPIc += 1;
//					adjRIc += 1;
				}
				if (denP != 0.0){
					adjPI[s] += (conAdjGI.getAdjTp() / denP);
//					adjPIc += 1;
				}
				if (denR != 0.0){
					adjRI[s] += (conAdjGI.getAdjTp() / denR);
//					adjRIc += 1;
				}
				
				
				// population model evaluation
				ArrowConfusionIS cong = new ArrowConfusionIS(trueBNI, GraphUtils.replaceNodes(graphP, trueBNI.getNodes()), context);
				AdjacencyConfusionIS conAdjG = new AdjacencyConfusionIS(trueBNI, GraphUtils.replaceNodes(graphP, trueBNI.getNodes()), context);

				// arr - nodes w/ CSI
				denP = (cong.getArrowsITp() + cong.getArrowsIFp());
				denR = (cong.getArrowsITp() + cong.getArrowsIFn());
				if (denP == 0.0 && denR == 0.0){
					arrIP[s] += 1.0;
					arrIR[s] += 1.0;
//					arrIPc += 1;
//					arrIRc += 1;
				}
				if (denP != 0.0){
					arrIP[s] += (cong.getArrowsITp() / denP);
//					arrIPc += 1;
				}
				if(denR != 0.0){
					arrIR[s] += (cong.getArrowsITp() / denR);
//					arrIRc += 1;
				}

				// arr - nodes w/o CSI
				denP = (cong.getArrowsNTp() + cong.getArrowsNFp());
				denR = (cong.getArrowsNTp() + cong.getArrowsNFn());
				if (denP == 0.0 && denR == 0.0){
					arrNP[s] += 1.0;
					arrNR[s] += 1.0;
//					arrNPc += 1;
//					arrNRc += 1;
				}
				if (denP != 0.0){
					arrNP[s] += (cong.getArrowsNTp() / denP);
//					arrNPc += 1;
				}
				if (denR != 0.0){
					arrNR[s] += (cong.getArrowsNTp() / denR);
//					arrNRc += 1;
				}

				// arr - over all nodes 
				denP = (cong.getArrowsTp() + cong.getArrowsFp());
				denR = (cong.getArrowsTp() + cong.getArrowsFn());
				if (denP == 0.0 && denR == 0.0){
					arrP[s] += 1.0;
					arrR[s] += 1.0;
//					arrPc += 1;
				}
				if (denP != 0.0){
					arrP[s] += (cong.getArrowsTp() / denP);
				}
				if (denR != 0.0){
					arrR[s] += (cong.getArrowsTp() / denR);
				}
				
				// adj - nodes w/ CSI
				denP = (conAdjG.getAdjITp() + conAdjG.getAdjIFp());
				denR = (conAdjG.getAdjITp() + conAdjG.getAdjIFn());
				if (denP == 0.0 && denR == 0.0){
					adjIP[s] += 1.0;
					adjIR[s] += 1.0;
				}
				if (denP != 0.0){
					adjIP[s] += (conAdjG.getAdjITp() / denP);
				}
				if(denR != 0.0){
					adjIR[s] += (conAdjG.getAdjITp() / denR);
				}
				
				// adj - nodes w/o CSI
				denP = (conAdjG.getAdjNTp() + conAdjG.getAdjNFp());
				denR = (conAdjG.getAdjNTp() + conAdjG.getAdjNFn());
				if (denP == 0.0 && denR == 0.0){
					adjNP[s] += 1.0;
					adjNR[s] += 1.0;
				}
				if (denP != 0.0){
					adjNP[s] += (conAdjG.getAdjNTp() / denP);
				}
				if (denR != 0.0){
					adjNR[s] += (conAdjG.getAdjNTp() / denR);
				}

				// adj - over all nodes 
				denP = (conAdjG.getAdjTp() + conAdjG.getAdjFp());
				denR = (conAdjG.getAdjTp() + conAdjG.getAdjFn());
				if (denP == 0.0 && denR == 0.0){
					adjP[s] += 1.0;
					adjR[s] += 1.0;
				}
				if (denP != 0.0){
					adjP[s] += (conAdjG.getAdjTp() / denP);
				}
				if (denR != 0.0){
					adjR[s] += (conAdjG.getAdjTp() / denR);
				}
				
				//				System.out.println("-------------");

				GraphUtils.GraphComparison cmpI = SearchGraphUtils.getGraphComparison(graphI, trueBNI);
				GraphUtils.GraphComparison cmpP = SearchGraphUtils.getGraphComparison(graphP, trueBNI);
				addedI[s] += cmpI.getEdgesAdded().size();
				removedI[s] += cmpI.getEdgesRemoved().size();
				reorientedI[s] += cmpI.getEdgesReorientedTo().size();

				added[s] += cmpP.getEdgesAdded().size();
				removed[s] += cmpP.getEdgesRemoved().size();
				reoriented[s] += cmpP.getEdgesReorientedTo().size();

				GraphUtils.GraphComparison cmpI2 = SearchGraphUtils.getGraphComparison(graphI, trueBNI, context);
				GraphUtils.GraphComparison cmpP2 = SearchGraphUtils.getGraphComparison(graphP, trueBNI, context);
				addedI_IS[s] += cmpI2.getEdgesAddedIS().size();
				removedI_IS[s] += cmpI2.getEdgesRemovedIS().size();
				reorientedI_IS[s] += cmpI2.getEdgesReorientedToIS().size();
				addedI_Other[s] += cmpI2.getEdgesAddedOther().size();
				removedI_Other[s] += cmpI2.getEdgesRemovedOther().size();
				reorientedI_Other[s] += cmpI2.getEdgesReorientedToOther().size();

				addedIS[s] += cmpP2.getEdgesAddedIS().size();
				removedIS[s] += cmpP2.getEdgesRemovedIS().size();
				reorientedIS[s] += cmpP2.getEdgesReorientedToIS().size();
				addedOther[s] += cmpP2.getEdgesAddedOther().size();
				removedOther[s] += cmpP2.getEdgesRemovedOther().size();
				reorientedOther[s] += cmpP2.getEdgesReorientedToOther().size();

				//DataSet data = DataUtils.concatenate(trainData, test);
				DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
				Graph dagI = iteratorI.next();
				llr[s] += fgesI.scoreDag(dagI) - fgesI.scoreDag(dagP);

			}
			avgcsi[s] /= (numVars * numTests);
			System.out.println("avgsci : "+ avgcsi[s]);

			arrIPI[s] /= numTests;
			arrIRI[s] /= numTests;
			arrNPI[s] /= numTests;
			arrNRI[s] /= numTests;
			arrPI[s] /= numTests;
			arrRI[s] /= numTests;
			adjIPI[s] /= numTests;
			adjIRI[s] /= numTests;
			adjNPI[s] /= numTests;
			adjNRI[s] /= numTests;
			adjPI[s] /= numTests;
			adjRI[s] /= numTests;
			
			addedI[s] /= numTests;
			removedI[s] /= numTests;
			reorientedI[s] /= numTests;

			addedI_IS[s] /= numTests;
			removedI_IS[s] /= numTests;
			reorientedI_IS[s] /= numTests;
			addedI_Other[s] /= numTests;
			removedI_Other[s] /= numTests;
			reorientedI_Other[s] /= numTests;
			//		llrI[s] /= numTests;

			arrIP[s] /= numTests;
			arrIR[s] /= numTests;
			arrNP[s] /= numTests;
			arrNR[s] /= numTests;
			arrP[s] /= numTests;
			arrR[s] /= numTests;
			adjIP[s] /= numTests;
			adjIR[s] /= numTests;
			adjNP[s] /= numTests;
			adjNR[s] /= numTests;
			adjP[s] /= numTests;
			adjR[s] /= numTests;
			
			added[s] /= numTests;
			removed[s] /= numTests;
			reoriented[s] /= numTests;
			
			addedIS[s] /= numTests;
			removedIS[s] /= numTests;
			reorientedIS[s] /= numTests;
			addedOther[s] /= numTests;
			removedOther[s] /= numTests;
			reorientedOther[s] /= numTests;
			llr[s] /= numTests;


		}
		//	printRes("CSI", numSim, adjIPI, adjNPI, adjPI, adjIRI, adjNRI, adjRI, addedI, removedI, reorientedI, llr);
		printRes(this.out, "CSI", numSim, arrIPI, arrNPI, arrPI, arrIRI, arrNRI, arrRI, adjIPI, adjNPI, adjPI, adjIRI, adjNRI, adjRI, addedI, removedI, reorientedI, addedI_IS, removedI_IS, reorientedI_IS, addedI_Other, removedI_Other, reorientedI_Other, llr);

		//		printRes("POP", numSim, adjIP, adjNP, adjP, adjIR, adjNR, adjR, added, removed, reoriented, llr);
		printRes(this.out,"POP", numSim, arrIP, arrNP, arrP, arrIR, arrNR, arrR, adjIP, adjNP, adjP, adjIR, adjNR, adjR, added, removed, reoriented, addedIS, removedIS, reorientedIS, addedOther, removedOther, reorientedOther, llr);
		this.out.close();
		System.out.println(avgOutDeg2/numSim);
		System.out.println(avgInDeg2/numSim);
		System.out.println("----------------------");

}

//	public void test1(){
//		int numVars = 4;
//		int numCases = 10000;
//		int minCat = 2;
//		int maxCat = 2;
//
//		List<Node> vars = new ArrayList<>();
//		//		for (int i = 0; i < numVars; i++) {
//		//			vars.add(new DiscreteVariable("A" + i));
//		//		}
//		vars.add(new DiscreteVariable("A"));
//		vars.add(new DiscreteVariable("B"));
//		vars.add(new DiscreteVariable("C"));
//		vars.add(new DiscreteVariable("D"));
//
//
//		Graph dag = new EdgeListGraph(vars);
//		dag.addDirectedEdge(dag.getNode("A"), dag.getNode("C"));
//		dag.addDirectedEdge(dag.getNode("B"), dag.getNode("C"));
//		dag.addDirectedEdge(dag.getNode("C"), dag.getNode("D"));
//		dag.addDirectedEdge(dag.getNode("B"), dag.getNode("D"));
//		BayesPm pm = new BayesPm(dag, minCat, maxCat);
//		MlBayesIm im = new MlBayesIm(pm, MlBayesIm.MANUAL);
//		im.setProbability(0, 0, 0, 0.75);
//		im.setProbability(0, 0, 1, 0.25);
//		im.setProbability(1, 0, 0, 0.62);
//		im.setProbability(1, 0, 1, 0.38);
//		im.setProbability(2, 0, 0, 0.92);
//		im.setProbability(2, 1, 0, 0.92);
//		im.setProbability(2, 2, 0, 0.31);
//		im.setProbability(2, 3, 0, 0.65);
//		im.setProbability(2, 0, 1, 0.08);
//		im.setProbability(2, 1, 1, 0.08);
//		im.setProbability(2, 2, 1, 0.69);
//		im.setProbability(2, 3, 1, 0.35);
//		im.setProbability(3, 0, 0, 0.6);
//		im.setProbability(3, 1, 0, 0.25);
//		im.setProbability(3, 2, 0, 0.1);
//		im.setProbability(3, 3, 0, 0.1);
//		im.setProbability(3, 0, 1, 0.4);
//		im.setProbability(3, 1, 1, 0.75);
//		im.setProbability(3, 2, 1, 0.9);
//		im.setProbability(3, 3, 1, 0.9);
//
//		System.out.println("IM:" + im);
//		DataSet data = im.simulateData(numCases, false);
//		DataSet test = im.simulateData(1, false);
//		test.setDouble(0, 0, 0);
//		test.setDouble(0, 1, 0);
//		test.setDouble(0, 2, 0);
//		test.setDouble(0, 3, 0);
//
//		BDeuScore popScore = new BDeuScore(data);
//		popScore.setSamplePrior(1);
//		Fges popFges = new Fges (popScore);
//		popFges.setVerbose(true);
//		Graph outP = popFges.search();
//		System.out.println("*************************************");
//
//		ISBDeuScore csi = new ISBDeuScore(data, test);
//		csi.setKAddition(0.5);
//		csi.setKDeletion(0.5);
//		csi.setKReorientation(0.5);
//		csi.setSamplePrior(1);
//		ISFges fgs = new ISFges(csi);
//		fgs.setPopulationGraph(SearchGraphUtils.chooseDagInPattern(outP));
//		fgs.setInitialGraph(SearchGraphUtils.chooseDagInPattern(outP));
//		fgs.setVerbose(true);
//		Graph out = fgs.search();
//		//		IndTestDSep pct= new IndTestDSep(dag);
//		//		pct.setVerbose(true);
//		//		Cpc pc = new Cpc(pct);
//		//		Graph pcg = pc.search();
//		//		System.out.println("pc facts: " + pct.getFacts());
//		System.out.println("test: " +test);
//		System.out.println("Dag: "+dag);
//		System.out.println("Pop: " + outP);//SearchGraphUtils.chooseDagInPattern(outP));
//		System.out.println("IS: " + out+"\n");//(SearchGraphUtils.chooseDagInPattern(out)));
//		System.out.println("PS_score = " + fgs.scoreDag(SearchGraphUtils.chooseDagInPattern(out))+"\n");
//		System.out.println("Pop_score = " + popFges.scoreDag(SearchGraphUtils.chooseDagInPattern(outP)));
//
//	}

	public void test3(){
		int numCases = 1000;
		int minCat = 2;
		int maxCat = 2;

		List<Node> vars = new ArrayList<>();
		//		for (int i = 0; i < numVars; i++) {
		vars.add(new DiscreteVariable("Y"));
		vars.add(new DiscreteVariable("Z"));
		vars.add(new DiscreteVariable("X"));

		//		}

		Graph dag = new EdgeListGraph(vars);
		dag.addDirectedEdge(dag.getNode("Y"), dag.getNode("X"));
		dag.addDirectedEdge(dag.getNode("Z"), dag.getNode("X"));

		BayesPm pm = new BayesPm(dag, minCat, maxCat);
		MlBayesIm im = new MlBayesIm(pm, MlBayesIm.MANUAL);
		im.setProbability(0, 0, 0, 0.75);
		im.setProbability(0, 0, 1, 0.25);
		im.setProbability(1, 0, 0, 0.51);
		im.setProbability(1, 0, 1, 0.49);
		im.setProbability(2, 0, 0, 0.9);
		im.setProbability(2, 0, 1, 0.1);
		im.setProbability(2, 1, 0, 0.9);
		im.setProbability(2, 1, 1, 0.1);
		im.setProbability(2, 2, 0, 0.23);
		im.setProbability(2, 2, 1, 0.77);
		im.setProbability(2, 3, 0, 0.52);
		im.setProbability(2, 3, 1, 0.48);


		System.out.println("IM:" + im);
		DataSet data = im.simulateData(numCases, false);
		DataSet test = im.simulateData(1, false);
		test.setDouble(0, 0, 0);
		test.setDouble(0, 1, 1);
		test.setDouble(0, 2, 1);

		BDeuScore popScore = new BDeuScore(data);
		Fges popFges = new Fges (popScore);
		Graph outP = popFges.search();
		ISBDeuScore csi = new ISBDeuScore(data, test);
		ISFges fgs = new ISFges(csi);
		fgs.setPopulationGraph(SearchGraphUtils.chooseDagInPattern(outP));
		//		fgs.setInitialGraph(SearchGraphUtils.chooseDagInPattern(outP));
		Graph out = fgs.search();
		
		popScore.localScore(2, new int[]{0,1});

		System.out.println("test: " +test);
		System.out.println("dag: "+dag);
		System.out.println("Pop: " + outP);
		System.out.println("IS: " + out + "\n");
		System.out.println("IS_score = " + fgs.scoreDag(SearchGraphUtils.chooseDagInPattern(out))+"\n");
		System.out.println("Pop_score = " + fgs.scoreDag(SearchGraphUtils.chooseDagInPattern(outP)));

	}
private void printRes(PrintStream out, String alg, int numSim, double[] arrIPI, double[] arrNPI, double[] arrPI, double[] arrIRI, double[] arrNRI, double[] arrRI, double[] adjIPI, double[] adjNPI, double[] adjPI, double[] adjIRI, double[] adjNRI, double[] adjRI, double[] addedI, double[] removedI, double[] reorientedI, double[] addedI_IS, double[] removedI_IS, double[] reorientedI_IS, double[] addedI_Other, double[] removedI_Other, double[] reorientedI_Other, double[] llrI){
	NumberFormat nf = new DecimalFormat("0.00");
	//		NumberFormat smallNf = new DecimalFormat("0.00E0");

	TextTable table = new TextTable(numSim+2, 8);
	table.setTabDelimited(true);
	String header = ", adj_P_IS, adj_P_NS, adj_P, adj_R_IS, adj_R_NS, adj_R, arr_P_IS, arr_P_NS, arr_P, arr_R_IS, arr_R_NS, arr_R, added_IS, added_NS, added, removed_IS, removed_NS, removed, reoriented_IS, reoriented_NS, reoriented, llr";
	table.setToken(0, 0, alg);
	table.setToken(0, 1, header);
	double arrIP = 0.0, arrNP = 0.0, arrP = 0.0, arrIR = 0.0, arrNR = 0.0, arrR = 0.0,
			adjIP = 0.0, adjNP = 0.0, adjP = 0.0, adjIR = 0.0, adjNR = 0.0, adjR = 0.0,
			added = 0.0, removed = 0.0, reoriented = 0.0,
			addedIS = 0.0, removedIS = 0.0, reorientedIS = 0.0,
			addedNS = 0.0, removedNS = 0.0, reorientedNS = 0.0, llr = 0.0;
	for (int i = 0; i < numSim; i++){
		String res = "," +nf.format(adjIPI[i])+","+nf.format(adjNPI[i])+","+nf.format(adjPI[i])+","+ nf.format(adjIRI[i])+
				","+nf.format(adjNRI[i])+","+nf.format(adjRI[i])+","+
				nf.format(arrIPI[i])+","+nf.format(arrNPI[i])+","+nf.format(arrPI[i])+","+ nf.format(arrIRI[i])+
				","+nf.format(arrNRI[i])+","+nf.format(arrRI[i])+","+
				nf.format(addedI_IS[i])+","+nf.format(addedI_Other[i])+","+nf.format(addedI[i])+","+
				nf.format(removedI_IS[i])+","+nf.format(removedI_Other[i])+","+nf.format(removedI[i])+","+
				nf.format(reorientedI_IS[i])+","+nf.format(reorientedI_Other[i])+","+nf.format(reorientedI[i])+","+ nf.format(llrI[i]);
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
		llr += llrI[i];
	}
	String res =  ","+nf.format(adjIP/numSim)+","+nf.format(adjNP/numSim)+","+nf.format(adjP/numSim)+","+nf.format(adjIR/numSim)+","+nf.format(adjNR/numSim)+","+nf.format(adjR/numSim)+","+
			nf.format(arrIP/numSim)+","+nf.format(arrNP/numSim)+","+nf.format(arrP/numSim)+","+nf.format(arrIR/numSim)+","+nf.format(arrNR/numSim)+","+nf.format(arrR/numSim)+","+
			nf.format(addedIS/numSim)+","+nf.format(addedNS/numSim)+","+nf.format(added/numSim)+","+
			nf.format(removedIS/numSim)+","+nf.format(removedNS/numSim)+","+nf.format(removed/numSim)+","+
			nf.format(reorientedIS/numSim)+","+nf.format(reorientedNS/numSim)+","+nf.format(reoriented/numSim)+","+nf.format(llr/numSim);
	table.setToken(numSim+1, 0, "avg");
	table.setToken(numSim+1, 1, res);
	out.println(table);
	System.out.println(table);		
}
//	private double getLikelihood(BayesIm im, DataSet dataSet) {
//
//		double lik = 0.0;
//
//		ROW:
//			for (int i = 0; i < dataSet.getNumRows(); i++) {
//				double lik0 = 0.0;
//
//				for (int j = 0; j < dataSet.getNumColumns(); j++) {
//int[] parents = im.getParents(j);
//int[] parentValues = new int[parents.length];
//
//for (int k = 0; k < parents.length; k++) {
//	parentValues[k] = dataSet.getInt(i, parents[k]);
//}
//
//int dataValue = dataSet.getInt(i, j);
//double p = im.getProbability(j, im.getRowIndex(j, parentValues), dataValue);
//
//if (p == 0) continue ROW;
//
//lik0 += Math.log(p);
//				}
//
//				lik += lik0;
//			}
//
//		return lik;
//	}


}
