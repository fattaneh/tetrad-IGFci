package edu.cmu.tetrad.test;

import java.util.ArrayList;
import java.util.List;

import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.ISMlBayesIm;
import edu.cmu.tetrad.data.ContinuousVariable;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphConverter;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.DagToPag;
import edu.cmu.tetrad.search.Fci;
import edu.cmu.tetrad.search.Fges;
import edu.cmu.tetrad.search.Pc;
import edu.cmu.tetrad.search.Rfci;
import edu.cmu.tetrad.search.IndTestDSep;
import edu.cmu.tetrad.search.IndependenceTest;

public class testPAG {
	public static void main(String[] args) {
//		Graph trueGraph = GraphConverter.convert("Latent(H),X1-->X0,X1-->X2,H-->X2,H-->X3,X3-->X4,X5-->X3");
//		Graph trueGraph = GraphConverter.convert("Latent(C),E-->B,C-->B,C-->D,A-->D,D-->F");
		Graph trueGraph = GraphConverter.convert("A-->B,B-->C,C-->D,A-->D,D-->F");
//		System.out.println(trueGraph);		
//
//		IndependenceTest test = new IndTestDSep(trueGraph);
////		test.setVerbose(true);
//		
//		Rfci fci = new Rfci(test);
//		fci.setVerbose(true);
//		Graph graph = fci.search();
//		System.out.println(graph);	
//
		Graph g1 = GraphConverter.convert("B-->C,C-->D,A-->D,D-->F");
		Graph g2 = GraphConverter.convert("A-->B,C-->D,A-->D,D-->F");
		Graph g3 = GraphConverter.convert("A-->B,B-->C,C-->D,A-->D,F-->D");
		BayesPm pm = new BayesPm(trueGraph, 2, 3);
		ISMlBayesIm im = new ISMlBayesIm(pm, ISMlBayesIm.RANDOM);
		DataSet data = im.simulateData(2000, false);
		System.out.println(trueGraph);		
		BDeuScore score = new BDeuScore(data);
//		test.setVerbose(true);
		Fges fci = new Fges(score);
		fci.setVerbose(true);
		Graph graph = fci.search();
		System.out.println(graph);	
		System.out.println(fci.scoreDag(trueGraph));
		System.out.println(fci.scoreDag(g1));
		System.out.println(fci.scoreDag(g2));
		System.out.println(fci.scoreDag(g3));

//
//		DagToPag dagToPag = new DagToPag(trueGraph);
//		Graph truePag = dagToPag.convert();
		

	}
}
