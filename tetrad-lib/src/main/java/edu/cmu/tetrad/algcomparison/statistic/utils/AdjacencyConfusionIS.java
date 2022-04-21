package edu.cmu.tetrad.algcomparison.statistic.utils;

import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Edges;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A confusion matrix for adjacencies--i.e. TP, FP, TN, FN for counts of adjacencies.
 *
 * @author Fattaneh
 */
public class AdjacencyConfusionIS {
    private Graph truth;
    private Graph est;
    private int adjTp;
    private int adjFp;
    private int adjFn;
    private int adjTn;
    private int adjITp;
    private int adjIFp;
    private int adjIFn;
    private int adjNTp;
    private int adjNFp;
    private int adjNFn;

    public AdjacencyConfusionIS(Graph truth, Graph est, Map<Node, Boolean> context) {
        this.truth = truth;
        this.est = est;
        adjTp = 0;
        adjFp = 0;
        adjFn = 0;
        adjITp = 0;
        adjIFp = 0;
        adjIFn = 0;
        adjNTp = 0;
        adjNFp = 0;
        adjNFn = 0;

        Set<Edge> allUnoriented = new HashSet<>();
        for (Edge edge : this.truth.getEdges()) {
            allUnoriented.add(Edges.undirectedEdge(edge.getNode1(), edge.getNode2()));
        }

        for (Edge edge : this.est.getEdges()) {
            allUnoriented.add(Edges.undirectedEdge(edge.getNode1(), edge.getNode2()));
        }

        for (Edge edge : allUnoriented) {
            if (this.est.isAdjacentTo(edge.getNode1(), edge.getNode2()) &&
                    !this.truth.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
            	if (context.get(truth.getNode(edge.getNode1().getName())) || context.get(truth.getNode(edge.getNode2().getName()))){
            		adjIFp++;
            	}
            	else{
            		adjNFp++;
            	}
        		adjFp++;

            }

            if (this.truth.isAdjacentTo(edge.getNode1(), edge.getNode2()) &&
                    !this.est.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
            	if (context.get(truth.getNode(edge.getNode1().getName())) || context.get(truth.getNode(edge.getNode2().getName()))){
            		adjIFn++;
            	}
            	else{
            		adjNFn++;
            	}
                adjFn++;
            }

            if (this.truth.isAdjacentTo(edge.getNode1(), edge.getNode2()) &&
                    this.est.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
            	if (context.get(truth.getNode(edge.getNode1().getName())) || context.get(truth.getNode(edge.getNode2().getName()))){
            		adjITp++;
            	}
            	else{
            		adjNTp++;
            	}
                adjTp++;
            }
        }

        int allEdges = this.truth.getNumNodes() * (this.truth.getNumNodes() - 1) / 2;

        adjTn = allEdges - adjFn;

//        System.out.println("adjTp: "+ adjTp);
//        System.out.println("adjTp I: "+ adjITp);
//        System.out.println("adjTp N: "+ adjNTp);
//        
//        System.out.println("adjFn: "+ adjFn);
//        System.out.println("adjFn I: "+ adjIFn);
//        System.out.println("adjFn N: "+ adjNFn);
//        
//        System.out.println("adjFp: "+ adjFp);
//        System.out.println("adjFp I: "+ adjIFp);
//        System.out.println("adjFp N: "+ adjNFp);
    }

    public int getAdjTp() {
        return adjTp;
    }
    
    public int getAdjITp() {
        return adjITp;
    }
    public int getAdjNTp() {
        return adjNTp;
    }
    
    public int getAdjFp() {
        return adjFp;
    }
    
    public int getAdjIFp() {
        return adjIFp;
    }
    
    public int getAdjNFp() {
        return adjNFp;
    }

    public int getAdjFn() {
        return adjFn;
    }

    public int getAdjIFn() {
        return adjIFn;
    }
    
    public int getAdjNFn() {
        return adjNFn;
    }
    public int getAdjTn() {
        return adjTn;
    }

}
