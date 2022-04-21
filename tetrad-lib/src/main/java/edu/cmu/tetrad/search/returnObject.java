package edu.cmu.tetrad.search;

import java.util.List;
import edu.cmu.tetrad.graph.Graph;

public class returnObject{
	public final List<Graph> instanceGraphs ;
	public final Graph populationGraph;
	public final double[][] probabilities;
	public returnObject(final List<Graph> instanceGraphs, final Graph populationGraph, double[][] probabilities) {
		this.instanceGraphs = instanceGraphs;
		this.populationGraph = populationGraph;
		this.probabilities = probabilities;
	}
}