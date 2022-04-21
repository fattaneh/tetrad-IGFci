package edu.cmu.tetrad.algcomparison.statistic.utils;

import edu.cmu.tetrad.graph.*;

import java.util.List;
import java.util.Map;

/**
 * A confusion matrix for arrows--i.e. TP, FP, TN, FN for counts of arrow endpoints.
 * A true positive arrow is counted for X*->Y in the estimated graph if X is not adjacent
 * to Y or X--Y or X<--Y.
 *
 * @author Fattaneh
 */
public class ArrowConfusionIS {

    // For arrowhead FP's, don't count an error unless the variables are adj in the true graph.
    private boolean truthAdj = false;

    private Graph truth;
    private Graph est;
    private int arrowsTp;
    private int arrowsTpc;
    private int arrowsFp;
    private int arrowsFpc;
    private int arrowsFn;
    private int arrowsFnc;
    private int arrowsTn;
    private int arrowsTnc;
    private int TCtp;
    private int TCfn;
    private int TCfp;
    private int arrowsITp;
    private int arrowsNTp;
    private int arrowsIFp;
    private int arrowsNFp;
    private int arrowsIFn;
    private int arrowsNFn;
    private int arrowsITn;
    private int arrowsNTn;

    public ArrowConfusionIS(Graph truth, Graph est, Map<Node, Boolean> context) {
        this(truth, est,  context, false);
    }

    public ArrowConfusionIS(Graph truth, Graph est, Map<Node, Boolean> context, boolean truthAdj) {
        this.truth = truth;
        this.est = est;
        arrowsTp = 0;
        arrowsTpc = 0;
        arrowsFp = 0;
        arrowsFpc = 0;
        arrowsFn = 0;
        arrowsFnc = 0;
        arrowsITp = 0;
        arrowsNTp = 0;
        arrowsIFp = 0;
        arrowsNFp = 0;
        arrowsIFn = 0;
        arrowsNFn = 0;
        arrowsITn = 0;
        arrowsNTn = 0;
        
        TCtp = 0; //for the two-cycle accuracy
        TCfn = 0;
        TCfp = 0;
        this.truthAdj = truthAdj;


        this.est = GraphUtils.replaceNodes(est, truth.getNodes());
        this.truth = GraphUtils.replaceNodes(truth, est.getNodes());


        // Get edges from the true Graph to compute TruePositives, TrueNegatives and FalseNeagtives
        //    System.out.println(this.truth.getEdges());

        for (Edge edge : this.truth.getEdges()) {

            List<Edge> edges1 = this.est.getEdges(edge.getNode1(), edge.getNode2());
            Edge edge1;

            if (edges1.size() == 1) {
                edge1 = edges1.get(0);
            } else {
                edge1 = this.est.getDirectedEdge(edge.getNode1(), edge.getNode2());
            }

            //      System.out.println(edge1 + "(est)");

            Endpoint e1Est = null;
            Endpoint e2Est = null;

            if (edge1 != null) {
                e1Est = edge1.getProximalEndpoint(edge.getNode1());
                e2Est = edge1.getProximalEndpoint(edge.getNode2());
            }
            //      System.out.println(e1Est);
            //      System.out.println(e2Est);

            List<Edge> edges2 = this.truth.getEdges(edge.getNode1(), edge.getNode2());
            Edge edge2;

            if (edges2.size() == 1) {
                edge2 = edges2.get(0);
//                if (Edges.isUndirectedEdge(edge2)) continue;
            } else {
                edge2 = this.truth.getDirectedEdge(edge.getNode1(), edge.getNode2());
            }

            //       System.out.println(edge2 + "(truth)");

            Endpoint e1True = null;
            Endpoint e2True = null;

            if (edge2 != null) {
                e1True = edge2.getProximalEndpoint(edge.getNode1());
                e2True = edge2.getProximalEndpoint(edge.getNode2());
            }
            //       System.out.println(e1True);
            //       System.out.println(e2True);


            if (e1True == Endpoint.ARROW && e1Est != Endpoint.ARROW) {
            	if (context.get(truth.getNode(edge.getNode1().getName()))){
            		arrowsIFn++;
            	}
            	else{
            		arrowsNFn++;
            	}
                arrowsFn++;
            }

            if (e2True == Endpoint.ARROW && e2Est != Endpoint.ARROW) {
            	if (context.get(truth.getNode(edge.getNode2().getName()))){
            		arrowsIFn++;
            	}
            	else{
            		arrowsNFn++;
            	}
                arrowsFn++;
            }

            if (e1True == Endpoint.ARROW && e1Est != Endpoint.ARROW && truth.isAdjacentTo(edge.getNode1(), edge.getNode2()) && est.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
                arrowsFnc = getArrowsFnc() + 1;
            }

            if (e2True == Endpoint.ARROW && e2Est != Endpoint.ARROW && truth.isAdjacentTo(edge.getNode1(), edge.getNode2()) && est.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
                arrowsFnc = getArrowsFnc() + 1;
            }


            if (e1True == Endpoint.ARROW && e1Est == Endpoint.ARROW) {
            	if (context.get(truth.getNode(edge.getNode1().getName()))){
            		arrowsITp++;
            	}
            	else{
            		arrowsNTp++;
            	}
                arrowsTp++;
            }

            if (e2True == Endpoint.ARROW && e2Est == Endpoint.ARROW) {
            	if (context.get(truth.getNode(edge.getNode2().getName()))){
            		arrowsITp++;
            	}
            	else{
            		arrowsNTp++;
            	}
                arrowsTp++;
            }

            if (e1True == Endpoint.ARROW && e1Est == Endpoint.ARROW && truth.isAdjacentTo(edge.getNode1(), edge.getNode2()) && est.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
                arrowsTpc = getArrowsTpc() + 1;
            }

            if (e2True == Endpoint.ARROW && e2Est == Endpoint.ARROW && truth.isAdjacentTo(edge.getNode1(), edge.getNode2()) && est.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
                arrowsTpc = getArrowsTpc() + 1;
            }

            if (e1True != Endpoint.ARROW && e1Est != Endpoint.ARROW) {
            	if (context.get(truth.getNode(edge2.getNode1().getName()))){
//            	if (context.get(truth.getNode(edge.getNode1().getName()))){
            		arrowsITn++;
            	}
            	else{
            		arrowsNTn++;
            	}
                arrowsTn++;
            }

            if (e2True != Endpoint.ARROW && e2Est != Endpoint.ARROW) {
            	if (context.get(truth.getNode(edge2.getNode2().getName()))){
            		arrowsITn++;
            	}
            	else{
            		arrowsNTn++;
            	}
                arrowsTn++;
            }

            if (e1True != Endpoint.ARROW && e1Est != Endpoint.ARROW && truth.isAdjacentTo(edge.getNode1(), edge.getNode2()) && est.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
                arrowsTnc = getArrowsTnc() + 1;
            }

            if (e2True != Endpoint.ARROW && e2Est != Endpoint.ARROW && truth.isAdjacentTo(edge.getNode1(), edge.getNode2()) && est.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
                arrowsTnc = getArrowsTnc() + 1;
            }
        }
// Get edges from the estimated graph to compute only FalsePositives
        // System.out.println(this.est.getEdges());

        for (Edge edge : this.est.getEdges()) {

            List<Edge> edges1 = this.est.getEdges(edge.getNode1(), edge.getNode2());
            Edge edge1;

            if (edges1.size() == 1) {
                edge1 = edges1.get(0);
            } else {
                edge1 = this.est.getDirectedEdge(edge.getNode1(), edge.getNode2());
            }
            //      System.out.println(edge1 + "(est)");

            Endpoint e1Est = null;
            Endpoint e2Est = null;

            if (edge1 != null) {
                e1Est = edge1.getProximalEndpoint(edge.getNode1());
                e2Est = edge1.getProximalEndpoint(edge.getNode2());
            }
            //       System.out.println(e1Est);
            //       System.out.println(e2Est);


            List<Edge> edges2 = this.truth.getEdges(edge.getNode1(), edge.getNode2());
            Edge edge2;

            if (edges2.size() == 1) {
                edge2 = edges2.get(0);
//                if (Edges.isUndirectedEdge(edge2)) continue;
            } else {
                edge2 = this.truth.getDirectedEdge(edge.getNode1(), edge.getNode2());
            }

            //          System.out.println(edge2 + "(truth)");

            Endpoint e1True = null;
            Endpoint e2True = null;

            if (edge2 != null) {
                e1True = edge2.getProximalEndpoint(edge.getNode1());
                e2True = edge2.getProximalEndpoint(edge.getNode2());
            }
            //          System.out.println(e1True);
            //          System.out.println(e2True);

//            if ((isTruthAdj() && truth.isAdjacentTo(edge.getNode1(), edge.getNode2()))) {
//                if (e1Est == Endpoint.ARROW && e1True!= Endpoint.ARROW) {
//                    arrowsFp++;
//                }
//
//                if (e2Est == Endpoint.ARROW && e2True!= Endpoint.ARROW) {
//                    arrowsFp++;
//                }
//            }

//            if (isTruthAdj()) {
//                if (truth.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
//                    if (e1Est == Endpoint.ARROW && e1True != Endpoint.ARROW) {
//                        arrowsFp++;
//                    }
//
//                    if (e2Est == Endpoint.ARROW && e2True != Endpoint.ARROW) {
//                        arrowsFp++;
//                    }
//                }
//            } else {
            if (e1Est == Endpoint.ARROW && e1True != Endpoint.ARROW) {
            	if (context.get(truth.getNode(edge1.getNode1().getName()))){
            		arrowsIFp++;
            	}
            	else{
            		arrowsNFp++;
            	}
            	arrowsFp++;
            }

            if (e2Est == Endpoint.ARROW && e2True != Endpoint.ARROW) {
            	if (context.get(truth.getNode(edge1.getNode2().getName()))){
            		arrowsIFp++;
            	}
            	else{
            		arrowsNFp++;
            	}
            	arrowsFp++;
            }
//            }

            if (e1Est == Endpoint.ARROW && e1True != Endpoint.ARROW && edge1 != null && edge2 != null) {
                arrowsFpc = getArrowsFpc() + 1;
            }

            if (e2Est == Endpoint.ARROW && e2True != Endpoint.ARROW && edge1 != null && edge2 != null) {
                arrowsFpc = getArrowsFpc() + 1;
            }

        }


        // test for 2-cycle
        //Set<Edge> allOriented = new HashSet<>();
        //allOriented.addAll(this.truth.getEdges());
        //allOriented.addAll(this.est.getEdges());

        for (Edge edge : this.truth.getEdges()) {


            List<Edge> TwoCycle1 = this.truth.getEdges(edge.getNode1(), edge.getNode2());
            List<Edge> TwoCycle2 = this.est.getEdges(edge.getNode1(), edge.getNode2());

            if (TwoCycle1.size() == 2 && TwoCycle2.size() == 2) {
                //              System.out.println("2-cycle correctly inferred " + TwoCycle1);
                TCtp++;
            }

            if (TwoCycle1.size() == 2 && TwoCycle2.size() != 2) {
                //             System.out.println("2-cycle not inferred " + TwoCycle1);
                TCfn++;
            }
        }

        for (Edge edge : this.est.getEdges()) {

            List<Edge> TwoCycle1 = this.truth.getEdges(edge.getNode1(), edge.getNode2());
            List<Edge> TwoCycle2 = this.est.getEdges(edge.getNode1(), edge.getNode2());

            if (TwoCycle1.size() != 2 && TwoCycle2.size() == 2) {
                //              System.out.println("2-cycle falsely inferred" + TwoCycle2);
                TCfp++;
            }
        }

  /*      System.out.println(arrowsTp);
        System.out.println(arrowsTn);
        System.out.println(arrowsFn);
        System.out.println(arrowsFp);
*/
        //divide by 2, the 2cycle accuracy is duplicated due to how getEdges is used
        TCtp = TCtp / 2;
        TCfn = TCfn / 2;
        TCfp = TCfp / 2;
        //       System.out.println(TCtp);
        //       System.out.println(TCfn);
        //       System.out.println(TCfp);
//      System.out.println("arrowsTp: "+ arrowsTp);
//      System.out.println("arrowsTp I: "+ arrowsITp);
//      System.out.println("arrowsTp N: "+ arrowsNTp);
//
//      System.out.println("arrowsTn: "+ arrowsTn);
//      System.out.println("arrowsTn I: "+ arrowsITn);
//      System.out.println("arrowsTn N: "+ arrowsNTn);
//      
//      System.out.println("arrowsFn: "+ arrowsFn);
//      System.out.println("arrowsFn I: "+ arrowsIFn);
//      System.out.println("arrowsFn N: "+ arrowsNFn);
//      
//      System.out.println("arrowsFp: "+ arrowsFp);
//      System.out.println("arrowsFp I: "+ arrowsIFp);
//      System.out.println("arrowsFp N: "+ arrowsNFp);
//
//      System.out.println("--------------------------");

    }


    public int getArrowsTp() {
        return arrowsTp;
    }
    
    public int getArrowsITp() {
        return arrowsITp;
    }
    
    public int getArrowsNTp() {
        return arrowsNTp;
    }

    public int getArrowsFp() {
        return arrowsFp;
    }

    public int getArrowsIFp() {
        return arrowsIFp;
    }

    public int getArrowsNFp() {
        return arrowsNFp;
    }

    public int getArrowsFn() {
        return arrowsFn;
    }

    public int getArrowsIFn() {
        return arrowsIFn;
    }
    public int getArrowsNFn() {
        return arrowsNFn;
    }
    
    public int getArrowsTn() {
        return arrowsTn;
    }
    
    public int getArrowsITn() {
        return arrowsITn;
    }
    
    public int getArrowsNTn() {
        return arrowsNTn;
    }


    public int getTwoCycleTp() {
        return TCtp;
    }

    public int getTwoCycleFp() {
        return TCfp;
    }

    public int getTwoCycleFn() {
        return TCfn;
    }

    /**
     * Two positives for common edges.
     */
    public int getArrowsTpc() {
        return arrowsTpc;
    }

    /**
     * False positives for common edges.
     */
    public int getArrowsFpc() {
        return arrowsFpc;
    }

    /**
     * False negatives for common edges.
     */
    public int getArrowsFnc() {
        return arrowsFnc;
    }

    /**
     * True Negatives for common edges.
     */
    public int getArrowsTnc() {
        return arrowsTnc;
    }

    public boolean isTruthAdj() {
        return truthAdj;
    }
}
