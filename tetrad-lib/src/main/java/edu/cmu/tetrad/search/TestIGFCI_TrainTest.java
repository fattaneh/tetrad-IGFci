package edu.cmu.tetrad.search;

import java.io.File;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import edu.cmu.tetrad.search.GFci;
import edu.cmu.tetrad.search.IGFci;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.ISBDeuScore;
import edu.cmu.tetrad.search.IndTestProbabilisticBDeu2;
import edu.cmu.tetrad.search.IndTestProbabilisticISBDeu2;
import edu.cmu.tetrad.search.SearchGraphUtils;
import edu.cmu.tetrad.util.DataConvertUtils;
import edu.cmu.tetrad.util.DelimiterUtils;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader;
import edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetReader;

class Key_GFCI {

	public final int n_a;
	public final int n_d;
	public final int n_r;


	public Key_GFCI(final int n_a, final int n_d, final int n_r) {
		this.n_a = n_a;
		this.n_d = n_d;
		this.n_r = n_r;
	}
	@Override
	public boolean equals (final Object O) {
		if (!(O instanceof Key_GFCI)) return false;
		if (((Key_GFCI) O).n_a != n_a) return false;
		if (((Key_GFCI) O).n_d != n_d) return false;
		if (((Key_GFCI) O).n_r != n_r) return false;
		return true;
	}
	 @Override
	 public int hashCode() {
		 return this.n_a ^ this.n_d ^ this.n_r ;
	 }
	 public String print(Key_GFCI key){
		return "("+key.n_a +", "+ key.n_d +", "+ key.n_r + ")";
	 }

}
public class TestIGFCI_TrainTest {

	public static void main(String[] args) {

		boolean threshold = true;
		double cutoff = 0.5;
		double samplePrior = 1.0;
		String data_path =  System.getProperty("user.dir"); 
		String pathToFolder = data_path + "/Shyam-data/";
		String dataName = "port_uni40";
		String pathToTrainData = pathToFolder + "PORT/" + dataName + "_train.csv";
		String pathToTestData = pathToFolder + "PORT/" + dataName + "_test.csv";
		String target = "217.DIREOUT";
		
		
		// Read in the training and test data
		DataSet trainData = readData(pathToTrainData);
		System.out.println(trainData.getNumRows() +", " + trainData.getNumColumns());

		DataSet testData = readData(pathToTestData);
		System.out.println(testData.getNumRows() +", " + testData.getNumColumns());


		// learn the population-wide model using the GFCi method and training data
		// define the BSC test
		IndTestProbabilisticBDeu2 indTest_pop = new IndTestProbabilisticBDeu2(trainData, 0.5);
		indTest_pop.setThreshold(threshold);
		indTest_pop.setCutoff(cutoff);
		indTest_pop.setSampleprior(samplePrior);
		// define the score 
		BDeuScore scoreP = new BDeuScore(trainData);
		scoreP.setSamplePrior(samplePrior);
		// run GFci
		GFci fci_pop = new GFci(indTest_pop, scoreP);
		Graph graphP = fci_pop.search();
		graphP = GraphUtils.replaceNodes(graphP, trainData.getVariables());

		PrintStream logFile;
		try {
			File dir = new File( pathToFolder + "/IGFCI/" + dataName);
			dir.mkdirs();
			String outputFileName = dataName  + "-PESS" + samplePrior + "_log.txt";
			File fileAUC = new File(dir, outputFileName);
			logFile = new PrintStream(new FileOutputStream(fileAUC));

		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		
		logFile.println(trainData.getNumRows() +", " + trainData.getNumColumns());
		logFile.println("Pop graph:" + graphP.getEdges());
		
		// run IGFci with different kappa's to learn instance-specific models  
		double[] kappa = new double[]{0.001, 0.1, 0.5,0.9};
		for (int p = 0; p < kappa.length; p++){	
			double k_add = kappa[p]; 
			double average_llr = 0.0;

			Map <Key_GFCI, Double> stats= new HashMap<Key_GFCI, Double>();
			PrintStream outForAUC;
			try {
				File dir = new File( pathToFolder + "/outputs/" + dataName);
				dir.mkdirs();
				String outputFileName = dataName + "-Kappa"+ k_add  +".csv";
				File fileAUC = new File(dir, outputFileName);
				outForAUC = new PrintStream(new FileOutputStream(fileAUC));

			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			
			System.out.println("kappa = " + k_add);
			logFile.println("kappa = " + k_add);
			

			outForAUC.println("y, population-FGES, instance-specific-FGES");
			double added = 0.0, removed = 0.0, reoriented = 0.0, shdStrict = 0.0, shdLenient = 0.0;

			// loop over the test set to learn an instance-specific PAG for each sample
			for (int i = 0; i < testData.getNumRows(); i++){
				DataSet test = testData.subsetRows(new int[]{i});
				
				// learn the instance-specific model using the IGFCi method, training data, and test
				// define the instance-specific BSC test
				IndTestProbabilisticISBDeu2 indTest_IS = new IndTestProbabilisticISBDeu2(trainData, test, indTest_pop.getH(), graphP);
				indTest_IS.setThreshold(threshold);
				indTest_IS.setCutoff(cutoff);
				indTest_IS.setSampleprior(samplePrior);
		
				// define the instance-specific score
				ISBDeuScore scoreI = new ISBDeuScore(trainData, test);
				scoreI.setKAddition(k_add);
				scoreI.setKDeletion(k_add);
				scoreI.setKReorientation(k_add);
				scoreI.setSamplePrior(samplePrior);
				
				//run IGFci
				IGFci Fci_IS = new IGFci(indTest_IS, scoreI, fci_pop.FgesGraph);
				Graph graphI = Fci_IS.search();
				graphI = GraphUtils.replaceNodes(graphI, trainData.getVariables());

				// compare the instance-specific and population-wide models
				GraphUtils.GraphComparison cmp = SearchGraphUtils.getGraphComparison(graphI, graphP, true);
				added += cmp.getEdgesAdded().size();
				removed += cmp.getEdgesRemoved().size();
				reoriented += cmp.getEdgesReorientedTo().size();
				shdStrict += cmp.getShdStrict();
				shdLenient += cmp.getShdLenient();

				int n_a = cmp.getEdgesAdded().size();
				int n_d = cmp.getEdgesRemoved().size();
				int n_r = cmp.getEdgesReorientedFrom().size();

				Key_GFCI cur_key = new Key_GFCI(n_a, n_d, n_r);
				if(stats.get(cur_key)!=null)
					stats.put(cur_key, stats.get(cur_key) + 1.0);
				else
					stats.put(cur_key, 1.0);

			}
		
			for (Key_GFCI k : stats.keySet()){
				System.out.println(k.print(k) + ":" + (stats.get(k)/testData.getNumRows())*100);
				logFile.println(k.print(k) + ":" + (stats.get(k)/testData.getNumRows())*100);

			}
			// compute average statistics and write the results in the output file
			added /= testData.getNumRows();
			removed /= testData.getNumRows();
			reoriented /= testData.getNumRows();
			shdStrict /= testData.getNumRows();
			shdLenient /= testData.getNumRows();
			System.out.println("average_llr: " + (average_llr/ testData.getNumRows()));
			logFile.println("avg added: " + added);
			logFile.println("avg removed: " + removed);
			logFile.println("avg reoriented: " + reoriented);
			logFile.println("avg shdStrict: " + shdStrict);
			logFile.println("avg shdLenient: " + shdLenient);

			System.out.println("-----------------");
			logFile.println("-----------------");
			outForAUC.close();
		}
		logFile.close();

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
