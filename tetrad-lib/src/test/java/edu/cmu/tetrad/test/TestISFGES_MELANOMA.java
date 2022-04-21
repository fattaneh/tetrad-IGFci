package edu.cmu.tetrad.test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.IKnowledge;
import edu.cmu.tetrad.data.Knowledge2;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.pitt.dbmi.data.reader.tabular.TabularDataReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;

public class TestISFGES_MELANOMA {
	public static void main(String[] args) {

		String workingDirectory = System.getProperty("user.dir");
		System.out.println(workingDirectory);
		
 		Path trainDataFile = Paths.get("/Users/fattanehjabbari/CCD-Project/CS-BN/Xinghua's_Data/gsva_dis_20.csv");

// 		Path trainDataFile = Paths.get("/Users/fattanehjabbari/CCD-Project/CS-BN/TDI_DEG/DEGmatrix.UPMCcell4greg.TDIDEGfeats.csv");
//		Path trainDataFile = Paths.get("/Users/fattanehjabbari/CCD-Project/CS-BN/TDI_DEG/DEGmatrix.UPMCcell4greg.allDEGfeats.csv");

		char delimiter = ',';

		VerticalDiscreteTabularDatasetReader trainDataReader = new VerticalDiscreteTabularDatasetFileReader(trainDataFile, DelimiterUtils.toDelimiter(delimiter));
		DataSet trainDataOrig = null;
		try {
			trainDataOrig = (DataSet) DataConvertUtils.toDataModel(trainDataReader.readInData());
			System.out.println(trainDataOrig.getNumRows() +", " + trainDataOrig.getNumColumns());
		} catch (Exception IOException) {
			IOException.printStackTrace();
		}
		int numVars = trainDataOrig.getNumColumns();
		IKnowledge knowledge = new Knowledge2();
		int[] tiers = new int[2];
		tiers[0] = 0;
		tiers[1] = 1;
		for (int i=0 ; i< numVars; i++) {
			if (!trainDataOrig.getVariable(i).getName().equals("PD1response")){
				knowledge.addToTier(0, trainDataOrig.getVariable(i).getName());
			}
			else{
				knowledge.addToTier(1, trainDataOrig.getVariable(i).getName());
			}
		}
		knowledge.setTierForbiddenWithin(0, true);

		// learn the population model
		BDeuScore scoreP = new BDeuScore(trainDataOrig);
		double structPrior = 1.0;
		scoreP.setStructurePrior(structPrior);
		Fges fgesP = new Fges (scoreP);
		fgesP.setKnowledge(knowledge);
//		fgesP.setSymmetricFirstStep(false);
		Graph graphP = fgesP.search();
		graphP = GraphUtils.replaceNodes(graphP, trainDataOrig.getVariables());
		System.out.println("Pop graph:" + graphP.getEdges());

		// estimate MAP parameters from the population model
		DagInPatternIterator iterator = new DagInPatternIterator(graphP);
		Graph dagP = iterator.next();
		dagP = GraphUtils.replaceNodes(dagP, trainDataOrig.getVariables());
		
		BayesPm pmP = new BayesPm(dagP);

		DirichletBayesIm priorP = DirichletBayesIm.symmetricDirichletIm(pmP, 1.0);
		BayesIm imP = DirichletEstimator.estimate(priorP, trainDataOrig);

		for (int p = 1; p <= 10; p++){
			
			double k_add =  p/10.0; //Math.pow(10, -1.0*p);
			double k_delete = k_add;
			double k_reverse = k_add;
			System.out.println("kappa = " + k_add);

			double[] llr = new double[trainDataOrig.getNumRows()];
			double[] probs_i = new double[trainDataOrig.getNumRows()];
			double[] probs_p = new double[trainDataOrig.getNumRows()];
			int[] truth = new int[trainDataOrig.getNumRows()];

			double average = 0.0;
			PrintStream out;
			PrintStream outForAUC;
			try {
				File dir = new File("/Users/fattanehjabbari/CCD-Project/CS-BN/TDI-DEG/");
				dir.mkdirs();
				String outputFileName = "MelanomaTDI-DEG-Distribution-kadd" + k_add+"-kdel" + k_delete+"-krev" + k_reverse+".csv";
//				String outputFileName = "MelanomaALL-DEG-Distribution-kadd" + k_add+"-kdel" + k_delete+"-krev" + k_reverse+".csv";

				File file = new File(dir, outputFileName);
				out = new PrintStream(new FileOutputStream(file));
				outputFileName = "MelanomaTDI-AUROC-kadd" + k_add+"-kdel" + k_delete+"-krev" + k_reverse+".csv";
//				outputFileName = "MelanomaALL-AUROC-kadd" + k_add+"-kdel" + k_delete+"-krev" + k_reverse+".csv";

				File fileAUC = new File(dir, outputFileName);
				outForAUC = new PrintStream(new FileOutputStream(fileAUC));
			} catch (Exception e) {
				throw new RuntimeException(e);
			}

			Map <Key, Double> stats= new HashMap<Key, Double>();
			Map <String, Double> DEGdist= new HashMap<String, Double>();
			for (int i = 0; i < trainDataOrig.getNumColumns()-1; i++){
				DEGdist.put(trainDataOrig.getVariable(i).getName(), 0.0);
			}

			outForAUC.println("PD1response, population-FGES, instance-specific-FGES");//, DEGs");
			out.println("DEGs, fraction of occurance in cases");

			for (int i = 0; i < trainDataOrig.getNumRows(); i++){
				DataSet trainData = trainDataOrig.copy();
				DataSet test = trainDataOrig.subsetRows(new int[]{i});
				trainData.removeRows(new int[]{i});
//				// learn the population model
//				BDeuScore scorePi = new BDeuScore(trainData);
//				scoreP.setStructurePrior(structPrior);
//				Fges fgesPi = new Fges (scorePi);
//				fgesPi.setKnowledge(knowledge);
//				Graph graphPi = fgesPi.search();
//				graphPi = GraphUtils.replaceNodes(graphPi, trainData.getVariables());
//				System.out.println("Pop graph "+ i + " :" + graphPi.getEdges());
				
				// learn the instance-specific model
				ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
				graphP = GraphUtils.replaceNodes(graphP, trainData.getVariables());

				scoreI.setKAddition(k_add);
				scoreI.setKDeletion(k_delete);
				scoreI.setKReorientation(k_reverse);
				scoreI.setStructurePrior(structPrior);
				ISFges fgesI = new ISFges(scoreI);
				fgesI.setKnowledge(knowledge);
				fgesI.setPopulationGraph(dagP);
				fgesI.setInitialGraph(dagP);
				Graph graphI = fgesI.search();
				graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());
				GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graphI, graphP);
				int n_a = cmp.getEdgesAdded().size();
				int n_d = cmp.getEdgesRemoved().size();
				Key cur_key = new Key(n_a, n_d);
				if(stats.get(cur_key)!=null)
					stats.put(cur_key, stats.get(cur_key) + 1.0);
				else
					stats.put(cur_key, 1.0);
	
				// learn a pop model from data + test
				DagInPatternIterator iteratorI = new DagInPatternIterator(graphI);
				Graph dagI = iteratorI.next();
				dagI = GraphUtils.replaceNodes(dagI, trainData.getVariables());
				BayesPm pmI = new BayesPm(dagI);
	
				DirichletBayesIm priorI = DirichletBayesIm.symmetricDirichletIm(pmI, 1.0);
				BayesIm imI = DirichletEstimator.estimate(priorI, trainData);
				fgesI.setPopulationGraph(dagP);
				llr[i] = fgesI.scoreDag(dagI) - fgesI.scoreDag(dagP);

	
				int diseaseIndex_p = imP.getNodeIndex(imP.getNode("PD1response"));
				imP = DirichletEstimator.estimate(priorP, trainData);
				int[] parents_p = imP.getParents(diseaseIndex_p);
				Arrays.sort(parents_p);
				int[] values_p = new int[parents_p.length];
			
				for (int no = 0; no < parents_p.length; no++){
					values_p [no] = test.getInt(0, parents_p[no]);
//					System.out.println(test.getVariable(parents_p[no]));
//					System.out.println(imP.getNode(parents_p[no]));
				}
				
				double prob_p = imP.getProbability(diseaseIndex_p, imP.getRowIndex(diseaseIndex_p, values_p), 1);
				probs_p[i] = prob_p; 
				int diseaseIndex_i = imI.getNodeIndex(imI.getNode("PD1response"));
				truth [i] = test.getInt(0,diseaseIndex_i);

				int[] parents_i = imI.getParents(diseaseIndex_i);
				Arrays.sort(parents_i);
				int[] values_i = new int[parents_i.length];
			
				for (int no = 0; no < parents_i.length; no++){
					values_i [no] = test.getInt(0, parents_i[no]);
				}
				double prob_i = imI.getProbability(diseaseIndex_i, imI.getRowIndex(diseaseIndex_i, values_i), 1);
				probs_i[i] = prob_i; 
				List<Node> parents_i_list = new ArrayList<Node>();
				for (int no = 0; no < parents_i.length; no++){
					DEGdist.put(imI.getNode(parents_i[no]).getName(), DEGdist.get(imI.getNode(parents_i[no]).getName())+1.0);
					parents_i_list.add(imI.getNode(parents_i[no]));
				}
				
				outForAUC.println(test.getInt(0,diseaseIndex_i) +", " + prob_p + ", "+ prob_i);//+ ", " + parents_i_list.toString());

				average += llr[i];
			}
			for (Key k : stats.keySet()){
				System.out.println(k.print(k) + ":" + (stats.get(k)/trainDataOrig.getNumRows())*100);
			}
			
	        Map<String, Double> sortedDEGdist = sortByValue(DEGdist, false);

			for (String k : sortedDEGdist.keySet()){
				out.println(k + ", " + (DEGdist.get(k)/trainDataOrig.getNumRows()));
				
			}
			out.println(Arrays.toString(llr));
			out.close();
			outForAUC.close();
			System.out.println( "AUROC: "+ AUC.measure(truth, probs_i));
			
//			System.out.println(average/trainDataOrig.getNumRows());
		}
		System.out.println("-----------------");

	}
	private static Map<String, Double> sortByValue(Map<String, Double> dEGdist, final boolean order)
    {
        List<Entry<String, Double>> list = new LinkedList<>(dEGdist.entrySet());

        // Sorting the list based on values
        list.sort((o1, o2) -> order ? o1.getValue().compareTo(o2.getValue()) == 0
                ? o1.getKey().compareTo(o2.getKey())
                : o1.getValue().compareTo(o2.getValue()) : o2.getValue().compareTo(o1.getValue()) == 0
                ? o2.getKey().compareTo(o1.getKey())
                : o2.getValue().compareTo(o1.getValue()));
        return list.stream().collect(Collectors.toMap(Entry::getKey, Entry::getValue, (a, b) -> b, LinkedHashMap::new));

    }
}
