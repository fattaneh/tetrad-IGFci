//package edu.cmu.tetrad.test;
//
//import java.io.File;
//import java.io.FileOutputStream;
//import java.io.PrintStream;
//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;
//
//import edu.cmu.tetrad.bayes.BayesPm;
//import edu.cmu.tetrad.bayes.MlBayesIm;
//import edu.cmu.tetrad.data.DataSet;
//import edu.cmu.tetrad.data.DataUtils;
//import edu.cmu.tetrad.data.DiscreteVariable;
//import edu.cmu.tetrad.data.IKnowledge;
//import edu.cmu.tetrad.data.Knowledge2;
//import edu.cmu.tetrad.graph.Graph;
//import edu.cmu.tetrad.graph.GraphUtils;
//import edu.cmu.tetrad.graph.IndependenceFact;
//import edu.cmu.tetrad.graph.Node;
//import edu.cmu.tetrad.graph.NodeType;
//import edu.cmu.tetrad.search.BDeuScore;
//import edu.cmu.tetrad.search.Fci;
//import edu.cmu.tetrad.search.IndTestChiSquare;
//import edu.cmu.tetrad.search.IndTestDSep;
//import edu.cmu.tetrad.search.IndTestProbabilisticBic;
//import edu.cmu.tetrad.search.IndependenceTest;
//import edu.cmu.tetrad.search.SearchGraphUtils;
//import edu.cmu.tetrad.util.RandomUtil;
//
//class PRKey {
//	double precision;
//	double recall;
//	double shd;
//	public PRKey(double precision, double recall, double shd){
//		this.precision = precision;
//		this.recall = recall;
//		this.shd = shd;
//	}
//	@Override
//	public boolean equals (final Object O) {
//		if (!(O instanceof PRKey)) return false;
//		if (((PRKey) O).precision != precision) return false;
//		if (((PRKey) O).recall != recall) return false;
//		if (((PRKey) O).shd != shd) return false;
//
//		return true;
//	}
//	public String print(PRKey key){
//		return "("+key.precision +", "+ key.recall+", "+ key.shd + ")";
//	}
//}
//
//public class TestChiVsProb {
//	private PrintStream outChi2;
//	private PrintStream outBsc;
//	private PrintStream outChi2_calibration;
//	private PrintStream outBsc_calibration;
//
//	public static void main(String[] args) {
//		TestChiVsProb t = new TestChiVsProb();
//		t.test_sim();
//	}
//
//	public void test_sim(){
//
//		//RandomUtil.getInstance().setSeed(1454147770L);
//		int[] numVarss = new int[]{20};
//		double[] edgesPerNodes = new double[]{2.0, 3.0, 4.0, 5.0};
//		int numCases = 2000;
//		int minCat = 2;
//		int maxCat = 4;
//		int numSim = 1;
//		boolean threshold = true;
//		double latent = 0.1;	
//		double[] alphas = new double[]{0.05};//{0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1};
//		double[] thresholds = new double[]{0.5};//, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99};
//
//		for (int numVars: numVarss){
//			for (double edgesPerNode : edgesPerNodes){
//				final int numEdges = (int) (numVars * edgesPerNode);
//				int numLatents = (int) Math.floor(numVars * latent);
//
//				System.out.println("# nodes: " + numVars + ", # edges: "+ numEdges + ", # numLatents: "+ numLatents + ", # training: " + numCases );
//
//				// loop over simulations
//				for (int s = 0; s < numSim; s++){
//
//					try {
//						File dir = new File("/Users/fattanehjabbari/CCD-Project/CS-BN/simulation-BscVsChi2-Calib/");
//
//						dir.mkdirs();
//						String outputFileName = "Chi2-V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-Sim" + s + ".csv";
//						File file = new File(dir, outputFileName);
//						this.outChi2 = new PrintStream(new FileOutputStream(file));
//						outputFileName = "Bsc-V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-Sim" + s + ".csv";
//						file = new File(dir, outputFileName);
//						this.outBsc= new PrintStream(new FileOutputStream(file));
//						
//						outputFileName = "Cal-Chi2-V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-Sim" + s + ".csv";
//						file = new File(dir, outputFileName);
//						this.outChi2_calibration = new PrintStream(new FileOutputStream(file));
//						outputFileName = "Cal-Bsc-V"+numVars +"-E"+ edgesPerNode +"-L"+ latent + "-N" + numCases + "-Sim" + s + ".csv";
//						file = new File(dir, outputFileName);
//						this.outBsc_calibration = new PrintStream(new FileOutputStream(file));
//					} catch (Exception e) {
//						throw new RuntimeException(e);
//					}
//					RandomUtil.getInstance().setSeed(1454147770L + s*10000 );
//					System.out.println("simulation: " + s);
//					
//					// keep the PR and SHD values
//					HashMap <String, PRKey> chi2map = new HashMap<>();
//					HashMap <String, PRKey> bscmap = new HashMap<>();
//
//
//					// create variables
//					IKnowledge knowledge = new Knowledge2();
//					List<Node> vars = new ArrayList<>();
//					int[] tiers = new int[numVars];
//					for (int i = 0; i < numVars; i++) {
//						vars.add(new DiscreteVariable("X" + i));
//						tiers[i] = i;
//						knowledge.addToTier(i, "X" + i);
//					}
//
//					// generate true BN and its parameters
//					Graph trueBN = GraphUtils.randomGraphRandomForwardEdges(vars, numLatents, numEdges, 30, 15, 15, false, true);
//					System.out.println("Latent Variables: " + getLatents(trueBN));
//
//					BayesPm pm = new BayesPm(trueBN, minCat, maxCat);
//					MlBayesIm im = new MlBayesIm(pm, MlBayesIm.RANDOM);
//
//					// simulate train and test data from BN
//					DataSet fullTrainData = im.simulateData(numCases, true, tiers);
//					DataSet trainData = DataUtils.restrictToMeasured(fullTrainData);
//
//					// calculate the ground truth PAG
//					IndependenceTest dsep = new IndTestDSep(trueBN);
//					Fci fci = new Fci(dsep);
//					Graph truePag = fci.search();
//					truePag = GraphUtils.replaceNodes(truePag, trueBN.getNodes());
//
//
//					BDeuScore scoreP = new BDeuScore(trainData);
//					scoreP.setStructurePrior(0.001);
//					scoreP.setSamplePrior(10.0);
//					
//					this.outChi2.println("alpha, precision, recall, shd");
//					this.outChi2_calibration.println("p,truth");
//					// loop here over alpha values
//					for (int a = 0; a < alphas.length ; a ++){
//						double alpha = alphas[a];
////						System.out.println("alpha: " + alpha);
//
//						// learn the population model using Chi2
//						IndTestChiSquare indTestChi2 = new IndTestChiSquare(trainData, alpha);
////						GFci gfciChi2 = new GFci(indTestChi2, scoreP);
//						Fci gfciChi2 = new Fci(indTestChi2);
//						gfciChi2.setDepth(4);
//						Graph graphChi2 = gfciChi2.search();
//						//System.out.println("graphChi2: " +graphChi2);
//
//						// compute the truth values of the tests when running search with chi2
//						double tp = 0.0, fp = 0.0, fn = 0.0;
//						HashMap<IndependenceFact, Double> testsChi2 = indTestChi2.getH();
//						for(IndependenceFact f: testsChi2.keySet()){
//							List<Node> _z = new ArrayList<Node>();
//							for(Node nz : f.getZ()){
//								_z.add(truePag.getNode(nz.getName()));
//							}
//							//boolean indep = (pValue > alpha());
//							boolean truth = dsep.isIndependent(truePag.getNode(f.getX().getName()), truePag.getNode(f.getY().getName()), _z);
//							double truthVal = truth ? 1.0 : 0.0;
//							double predictionVal = (testsChi2.get(f) > alpha) ? 1.0 : 0.0;
////							System.out.println("truth: " + truth);
////							System.out.println("predictionVal: " + predictionVal);
//
//							this.outChi2_calibration.println(predictionVal + ","+ truthVal);
//							if (truth && (testsChi2.get(f) > alpha)){
//								tp += 1.0;
//							}
//							else if (truth && (testsChi2.get(f) <= alpha)){
//								fn +=1;
//							}
//							else if (!truth && (testsChi2.get(f) > alpha)){
//								fp += 1.0;
//							}
////					        System.out.println("--------------------");
//
//						}
//						double precision = tp / (tp + fp);
//						double recall = tp / (tp + fn);
//						GraphUtils.GraphComparison comparisonChi2 = SearchGraphUtils.getGraphComparison(graphChi2, truePag);
//						double shd = comparisonChi2.getShd();
//						String key = Double.toString(alpha);
//						PRKey val = new PRKey (precision, recall, shd);
//						chi2map.put(key, val);
//						this.outChi2.println(key + ", " + precision + ", " + recall + ",  " + shd);
//						System.out.println("alpha = " + key + ":    " + chi2map.get(key).print(chi2map.get(key)));
//
//					}
//
//					this.outBsc.println("threshold, precision, recall, shd");
//					this.outBsc_calibration.println("p,truth");
//					
//					// loop over threshold values
//					for (int c = 0; c < thresholds.length ; c ++){
//						double cutoff = thresholds[c];
////						System.out.println("cutoff: " + cutoff);
//
//						// learn the population model using Bsc
//						IndTestProbabilisticBic indTestBsc = new IndTestProbabilisticBic(trainData);
//						indTestBsc.setThreshold(threshold);
//						indTestBsc.setCutoff(cutoff);
//						Fci gfciBsc = new Fci(indTestBsc);
//						gfciBsc.setDepth(4);
////						GFci gfciBsc = new GFci(indTestBsc, scoreP);
//						Graph graphBsc = gfciBsc.search();
//
//						// compute the truth values of the tests when running search with bsc
//						double tp = 0.0, fp = 0.0, fn = 0.0;
//						Map<IndependenceFact, Double> testsBsc = indTestBsc.getH();
//						for(IndependenceFact f: testsBsc.keySet()){
//							List<Node> _z = new ArrayList<Node>();
//							for(Node nz : f.getZ()){
//								_z.add(truePag.getNode(nz.getName()));
//							}
//							boolean truth = dsep.isIndependent(truePag.getNode(f.getX().getName()), truePag.getNode(f.getY().getName()), _z);
////							System.out.println("truth: " + truth);
////							System.out.println("testsBsc.get(f) : " + testsBsc.get(f) );
//
//							double truthVal = truth ? 1.0 : 0.0;
//							this.outBsc_calibration.println(testsBsc.get(f) + "," + truthVal);
//							if (truth && (testsBsc.get(f) >= cutoff)){
//								tp += 1.0;
//							}
//							else if (truth && (testsBsc.get(f) < cutoff)){
//								fn +=1;
//							}
//							else if (!truth && (testsBsc.get(f) >= cutoff)){
//								fp += 1.0;
//							}
////					        System.out.println("--------------------");
//						}
//						double precision = tp / (tp + fp);
//						double recall = tp / (tp + fn);
//						GraphUtils.GraphComparison comparisonBsc = SearchGraphUtils.getGraphComparison(graphBsc, truePag);
//						double shd = comparisonBsc.getShd();
////						System.out.println("truePag: " + truePag);
////						System.out.println("graphBsc: " + graphBsc);
////						System.out.println("Added: " + comparisonBsc.getEdgesAdded());
////						System.out.println("removed: " + comparisonBsc.getEdgesRemoved());
////						System.out.println("reoriented: " + comparisonBsc.getEdgesReorientedFrom());
//
//						PRKey val = new PRKey (precision, recall, shd);
//						String key = Double.toString(cutoff); 
//						bscmap.put(key, val);
//						this.outBsc.println(key + ", " + precision + ", " + recall + ",  " + shd);
//						System.out.println("threshold = " + key + ":    " + bscmap.get(key).print(bscmap.get(key)));
//						System.out.println("----------------------");
//
//					}
//					this.outBsc.close();
//					this.outBsc_calibration.close();
//					this.outChi2.close();
//					this.outChi2_calibration.close();
//				}
//			}
//		}
//	}
//	private List<Node> getLatents(Graph dag) {
//		List<Node> latents = new ArrayList<>();
//		for (Node n : dag.getNodes()) {
//			if (n.getNodeType() == NodeType.LATENT) {
//				latents.add(n);
//			}
//		}
//		return latents;
//	}
//}
