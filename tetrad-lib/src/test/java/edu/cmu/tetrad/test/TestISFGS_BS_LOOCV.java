package edu.cmu.tetrad.test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;

public class TestISFGS_BS_LOOCV {
	public static void main(String[] args) {

		//		
		//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
		//		String dataName = "genims_mortality_4i";
		//		String pathToData = pathToFolder + "GenIMS/" + dataName + ".csv";
		//		String target = "day90_status";

//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
//		String dataName = "genims_sepsis_5i";
//		String pathToData = pathToFolder + "GenIMS/" + dataName + ".csv";
//		String target = "everss";

//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
//		String dataName = "port_all";
////		String pathToData = pathToFolder + "PORT/" + dataName + ".csv";
//		String target = "217.DIREOUT";

		//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
		//		String dataName = "heart_death_di_1ksample";
		//		String pathToData = pathToFolder + "Heart Failure/" + dataName + ".csv";
		//		String target = "death";

		//		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/Shyam-data/";
		//		String dataName = "heart_deathcomp_di_1ksample";
		//		String pathToData = pathToFolder + "Heart Failure/" + dataName + ".csv";
		//		String target = "deathcomp";

		// Read in the data
		
		String pathToFolder = "/Users/fattanehjabbari/CCD-Project/CS-BN/lung_cancer/data/";
		String dataName = "LCMR_Processed4_NA_surv";
		String pathToData = pathToFolder + dataName + ".csv";
		String target = "Survive1";
		
		// Read in the data
		DataSet trainDataOrig = readData(pathToData);

		int numBootstraps = 20;
		List<DataSet> bsSamples_train = new ArrayList<>(numBootstraps);

		// Generate the bootstrap samples
		for (int bootstrap = 0; bootstrap < numBootstraps; bootstrap++){
			DataSet bsData = DataUtils.getBootstrapSample(trainDataOrig, trainDataOrig.getNumRows());
			bsSamples_train.add(bsData);
			
		}
			// learn the population model using all training data
		double samplePrior = 20.0;
		double structurePrior = 1.0;

		for (int p = 1; p <= 10; p++){

			double k_add =  p/10.0; //Math.pow(10, -1.0*p);
			System.out.println("kappa = " + k_add);
			double[][] probs_is = new double[trainDataOrig.getNumRows()][numBootstraps];
			double[][] probs_p = new double[trainDataOrig.getNumRows()][numBootstraps];
			int[] truth = new int[trainDataOrig.getNumRows()];
			PrintStream outForAUC_Pop, outForAUC_IS;
			try {
				File dir = new File( pathToFolder + "/outputs/" + dataName);
				dir.mkdirs();
				String outputFileName = dataName + "_AUROC_Kappa"+ k_add +"_Pop.csv";
				File fileAUC = new File(dir, outputFileName);
				outForAUC_Pop = new PrintStream(new FileOutputStream(fileAUC));
				outputFileName = dataName + "_AUROC_Kappa"+ k_add +"_Is.csv";
				fileAUC = new File(dir, outputFileName);
				outForAUC_IS = new PrintStream(new FileOutputStream(fileAUC));
			} catch (Exception e) {
				throw new RuntimeException(e);
			}

			for (int bs = 0; bs < numBootstraps; bs ++){
				if (bs % 20==0)
					System.out.println("BS: " + bs);
				
				DataSet trainData = bsSamples_train.get(bs);
				IKnowledge knowledge = createKnowledge(trainData, target);
				Graph graphP = BNlearn_pop(trainData, knowledge, samplePrior, structurePrior);
				
				// loop over test cases
				for (int i = 0; i < trainDataOrig.getNumRows(); i++){
					DataSet test = trainDataOrig.subsetRows(new int[]{i});

					// learn the IS graph
					Graph graphI = learnBNIS(trainData, test, k_add, graphP, knowledge, samplePrior);

					// compute probability distribution of the target variable
					int targetIndex = trainData.getColumn(trainData.getVariable(target)); //imP.getNodeIndex(imP.getNode(target));
					truth[i] = test.getInt(0, targetIndex);

					//get the prob from IS model
					probs_is[i][bs] = estimation(trainData, test, graphI, target);

					//get the prob from population model
					double prob_p = estimation(trainData, test, graphP, target);
					probs_p[i][bs] = prob_p; 
				}
			}
			
			// write the results to output files
			// write the first line , i.e. header
			outForAUC_Pop.print("y");
			outForAUC_IS.print("y");

			for (int bs = 0; bs < numBootstraps; bs ++){
				outForAUC_Pop.print(",P_Pop_" + bs);
				outForAUC_IS.print(",P_IS_" + bs);
			}
			outForAUC_Pop.println();
			outForAUC_IS.println();

			// write the probs
			for (int i = 0; i < trainDataOrig.getNumRows(); i++){
				outForAUC_Pop.print(truth[i]);
				outForAUC_IS.print(truth[i]);
				for (int bs = 0; bs < numBootstraps; bs ++){
					outForAUC_Pop.print(","+ probs_p[i][bs]);
				    outForAUC_IS.print(","+ probs_is[i][bs]);
				}
				outForAUC_Pop.println();
				outForAUC_IS.println();
			}
			outForAUC_Pop.close();
			outForAUC_IS.close();
			System.out.println("-----------------");
		}
	}

	private static Graph learnBNIS(DataSet trainData, DataSet test, double kappa, Graph graphP, IKnowledge knowledge, double samplePrior){
		// learn the instance-specific model
		ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
		scoreI.setSamplePrior(samplePrior);
		scoreI.setKAddition(kappa);
		scoreI.setKDeletion(kappa);
		scoreI.setKReorientation(kappa);
		ISFges fgesI = new ISFges(scoreI);
		fgesI.setKnowledge(knowledge);
		fgesI.setPopulationGraph(graphP);
		fgesI.setInitialGraph(graphP);
		Graph graphI = fgesI.search();
		graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
		return graphI;
	}

	private static double estimation(DataSet trainData, DataSet test, Graph graph, String target){

		DagInPatternIterator iterator = new DagInPatternIterator(graph);
		Graph dag = iterator.next();
		dag = GraphUtils.replaceNodes(dag, trainData.getVariables());
		BayesPm pm = new BayesPm(dag);

		DirichletBayesIm prior = DirichletBayesIm.symmetricDirichletIm(pm, 1.0);
		BayesIm im = DirichletEstimator.estimate(prior, trainData);

		//		System.out.println("mbn: " + dag.getParents(dag.getNode(target)));	
		int targetIndex = im.getNodeIndex(im.getNode(target));
		int[] parents = im.getParents(targetIndex);
		Arrays.sort(parents);
		int[] values = new int[parents.length];
		for (int no = 0; no < parents.length; no++){
			values [no] = test.getInt(0, parents[no]);
		}
		double prob = im.getProbability(targetIndex, im.getRowIndex(targetIndex, values), 1);
		return prob;
	}
	//	private static double kappaTuning(DataSet trainData, IKnowledge knowledge, Graph dagP, String target){
	//		//inner LOOCV
	//		int kvalues = 9;
	//		double[] kappa_tune = new double[kvalues+1];
	//		double[] auroc_tune = new double[kvalues+1];
	//		for (int p = 0; p <= kvalues; p++){	
	//			double k_add =  (p+1)/10.0; 
	//			double k_delete = k_add;
	//			double k_reverse = k_add;
	//			kappa_tune[p]= k_add;
	//			System.out.println("kappa: " + k_add);
	//
	//			double[] probs_is_tune = new double[trainData.getNumRows()];
	//			int[] truth_tune = new int[trainData.getNumRows()];
	//			for (int j = 0; j < trainData.getNumRows(); j++){
	//				DataSet trainData66 = trainData.copy();
	//				DataSet test66 = trainData.subsetRows(new int[]{j});
	//				trainData66.removeRows(new int[]{j});
	//
	//				// learn the instance-specific model
	//				//				System.out.println("trainData66: "+trainData66.getNumRows());
	//				ISBDeuScore scoreI = new ISBDeuScore(trainData66, test66);
	//				scoreI.setKAddition(k_add);
	//				scoreI.setKDeletion(k_delete);
	//				scoreI.setKReorientation(k_reverse);
	//				ISFges fgesI = new ISFges(scoreI);
	//				fgesI.setKnowledge(knowledge);
	//				fgesI.setPopulationGraph(dagP);
	//				fgesI.setInitialGraph(dagP);
	//				Graph graphI = fgesI.search();
	//				graphI = GraphUtils.replaceNodes(graphI, trainData66.getVariables());
	//
	//				// learn a pop model from data + test
	//				DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
	//				Graph dagI = iteratorI.next();
	//				dagI = GraphUtils.replaceNodes(dagI, trainData66.getVariables());
	//				BayesPm pmI = new BayesPm(dagI);
	//
	//				DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
	//				BayesIm imI = DirichletEstimator.estimate(priorI, trainData66);
	//				fgesI.setPopulationGraph(dagP);
	//
	//				int diseaseIndex_i = imI.getNodeIndex(imI.getNode(target));
	//				truth_tune [j] = test66.getInt(0,diseaseIndex_i);
	//
	//				double prob_i = estimation_pop(imI, test66, diseaseIndex_i);
	//				probs_is_tune[j] = prob_i; 
	//			}
	//
	//			auroc_tune[p] = FCR.measure(truth_tune, probs_is_tune, 0.85, 0.15);
	//			//			System.out.println( "AUROC: "+ auroc_tune[p]);
	//		}
	//
	//		QuickSort.sort(auroc_tune, kappa_tune);
	//
	//
	//		double best_kappa = kappa_tune[kvalues];
	//		double best_auroc = auroc_tune[kvalues];
	//		if (best_auroc == 0.0){
	//			best_kappa = 0.3;
	//		}
	//		System.out.println( "BEST fcr: "+ best_auroc + ", BEST kappa: " + best_kappa);
	//		return best_kappa;
	//	}

	private static Graph BNlearn_pop(DataSet trainDataOrig, IKnowledge knowledge, double samplePrior, double structurePrior) {
		BDeuScore scoreP = new BDeuScore(trainDataOrig);
		scoreP.setSamplePrior(samplePrior);
		scoreP.setStructurePrior(structurePrior);
		Fges fgesP = new Fges (scoreP);
		fgesP.setKnowledge(knowledge);
		fgesP.setSymmetricFirstStep(true);
		Graph graphP = fgesP.search();
		graphP = GraphUtils.replaceNodes(graphP, trainDataOrig.getVariables());
		return graphP;
	}
	private static IKnowledge createKnowledge(DataSet trainDataOrig, String target) {
		int numVars = trainDataOrig.getNumColumns();
		IKnowledge knowledge = new Knowledge2();
		int[] tiers = new int[2];
		tiers[0] = 0;
		tiers[1] = 1;
		for (int i=0 ; i< numVars; i++) {
			if (!trainDataOrig.getVariable(i).getName().equals(target)){
				knowledge.addToTier(0, trainDataOrig.getVariable(i).getName());
			}
			else{
				knowledge.addToTier(1, trainDataOrig.getVariable(i).getName());
			}
		}
		knowledge.setTierForbiddenWithin(0, true);
		return knowledge;
	}
	private static DataSet readData(String pathToData) {
		Path trainDataFile = Paths.get(pathToData);
		char delimiter = ',';
		VerticalDiscreteTabularDatasetReader trainDataReader = new VerticalDiscreteTabularDatasetFileReader(trainDataFile, DelimiterUtils.toDelimiter(delimiter));
		DataSet trainDataOrig = null;
		try {
			trainDataOrig = (DataSet) DataConvertUtils.toDataModel(trainDataReader.readInData());
		} catch (Exception IOException) {
			IOException.printStackTrace();
		}
		return trainDataOrig;
	}
	
}