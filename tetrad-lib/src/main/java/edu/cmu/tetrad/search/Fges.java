///////////////////////////////////////////////////////////////////////////////
// For information as to what this class does, see the Javadoc, below.       //
// Copyright (C) 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,       //
// 2007, 2008, 2009, 2010, 2014, 2015 by Peter Spirtes, Richard Scheines, Joseph   //
// Ramsey, and Clark Glymour.                                                //
//                                                                           //
// This program is free software; you can redistribute it and/or modify      //
// it under the terms of the GNU General Public License as published by      //
// the Free Software Foundation; either version 2 of the License, or         //
// (at your option) any later version.                                       //
//                                                                           //
// This program is distributed in the hope that it will be useful,           //
// but WITHOUT ANY WARRANTY; without even the implied warranty of            //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
// GNU General Public License for more details.                              //
//                                                                           //
// You should have received a copy of the GNU General Public License         //
// along with this program; if not, write to the Free Software               //
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA //
///////////////////////////////////////////////////////////////////////////////
package edu.cmu.tetrad.search;

import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.util.*;

import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;
import java.util.concurrent.*;

/**
 * GesSearch is an implementation of the GES algorithm, as specified in
 * Chickering (2002) "Optimal structure identification with greedy search"
 * Journal of Machine Learning Research. It works for both BayesNets and SEMs.
 * <p>
 * Some code optimization could be done for the scoring part of the graph for
 * discrete models (method scoreGraphChange). Some of Andrew Moore's approaches
 * for caching sufficient statistics, for instance.
 * <p>
 * To speed things up, it has been assumed that variables X and Y with zero
 * correlation do not correspond to edges in the graph. This is a restricted
 * form of the heuristicSpeedup assumption, something GES does not assume. This
 * the graph. This is a restricted form of the heuristicSpeedup assumption,
 * something GES does not assume. This heuristicSpeedup assumption needs to be
 * explicitly turned on using setHeuristicSpeedup(true).
 * <p>
 * A number of other optimizations were added 5/2015. See code for details.
 *
 * @author Ricardo Silva, Summer 2003
 * @author Joseph Ramsey, Revisions 5/2015
 */
public final class Fges implements GraphSearch, GraphScorer {

    /**
     * Internal.
     */

    private enum Mode {
        allowUnfaithfulness, heuristicSpeedup, coverNoncolliders
    }

    /**
     * Specification of forbidden and required edges.
     */
    private IKnowledge knowledge = new Knowledge2();

    /**
     * List of variables in the data set, in order.
     */
    private List<Node> variables;

    /**
     * The true graph, if known. If this is provided, asterisks will be printed
     * out next to false positive added edges (that is, edges added that aren't
     * adjacencies in the true graph).
     */
    private Graph trueGraph;

    /**
     * An initial graph to start from.
     */
    private Graph initialGraph;

    /**
     * If non-null, edges not adjacent in this graph will not be added.
     */
    private Graph boundGraph = null;

    /**
     * Elapsed time of the most recent search.
     */
    private long elapsedTime;

    /**
     * A bound on cycle length.
     */
    private int cycleBound = -1;

    /**
     * The totalScore for discrete searches.
     */
    private Score score;

    /**
     * The logger for this class. The config needs to be set.
     */
    private TetradLogger logger = TetradLogger.getInstance();

    /**
     * The top n graphs found by the algorithm, where n is numPatternsToStore.
     */
    private LinkedList<ScoredGraph> topGraphs = new LinkedList<>();

    /**
     * True if verbose output should be printed.
     */
    private boolean verbose = false;

    // Potential arrows sorted by bump high to low. The first one is a candidate for adding to the graph.
    private SortedSet<Arrow> sortedArrows = null;

    // Arrows added to sortedArrows for each <i, j>.
    private Map<OrderedPair<Node>, Set<Arrow>> lookupArrows = null;

    // A utility map to help with orientation.
    private Map<Node, Set<Node>> neighbors = null;

    // Map from variables to their column indices in the data set.
    private ConcurrentMap<Node, Integer> hashIndices;

    // The static ForkJoinPool instance.
    private final ForkJoinPool pool;

    // A graph where X--Y means that X and Y have non-zero total effect on one another.
    private Graph effectEdgesGraph;

    // Where printed output is sent.
    private PrintStream out = System.out;

    // A initial adjacencies graph.
    private Graph adjacencies = null;

    // The graph being constructed.
    private Graph graph;

    // Arrows with the same totalScore are stored in this list to distinguish their order in sortedArrows.
    // The ordering doesn't matter; it just have to be transitive.
    private int arrowIndex = 0;

    // The BIC score of the model.
    private double modelScore;

    // Internal.
    private Mode mode = Mode.heuristicSpeedup;

    /**
     * True if one-edge faithfulness is assumed. Speedse the algorithm up.
     */
    private boolean faithfulnessAssumed = true;

    // Bounds the degree of the graph.
    private int maxDegree = -1;

    // True if the first step of adding an edge to an empty graph should be scored in both directions
    // for each edge with the maximum score chosen.
    private boolean symmetricFirstStep = false;

    // The maximum number of threads to use.
    private final int maxThreads;

    //===========================CONSTRUCTORS=============================//

    /**
     * Construct a Score and pass it in here. The totalScore should return a
     * positive value in case of conditional dependence and a negative values in
     * case of conditional independence. See Chickering (2002), locally
     * consistent scoring criterion. This by default uses all of the processors on
     * the machine.
     */
    public Fges(Score score) {
        this(score, Runtime.getRuntime().availableProcessors());
    }

    /**
     * Lets one construct with a score and a parallelism, that is, the number of threads to effectively use.
     */
    public Fges(Score score, int parallelism) {
        if (score == null) {
            throw new NullPointerException();
        }
        setScore(score);
        this.maxThreads = parallelism;
        this.pool = new ForkJoinPool(parallelism);
        this.graph = new EdgeListGraphSingleConnections(getVariables());
    }

    //==========================PUBLIC METHODS==========================//

    /**
     * Set to true if it is assumed that all path pairs with one length 1 path
     * do not cancel.
     */
    public void setFaithfulnessAssumed(boolean faithfulnessAssumed) {
        this.faithfulnessAssumed = faithfulnessAssumed;
    }

    /**
     * @return true if it is assumed that all path pairs with one length 1 path
     * do not cancel.
     */
    public boolean isFaithfulnessAssumed() {
        return faithfulnessAssumed;
    }

    /**
     * Greedy equivalence search: Start from the empty graph, add edges till
     * model is significant. Then start deleting edges till a minimum is
     * achieved.
     *
     * @return the resulting Pattern.
     */
    public Graph search() {
        long start = System.currentTimeMillis();
        topGraphs.clear();

        lookupArrows = new ConcurrentHashMap<>();
        final List<Node> nodes = new ArrayList<>(variables);
        graph = new EdgeListGraphSingleConnections(nodes);

        if (adjacencies != null) {
            adjacencies = GraphUtils.replaceNodes(adjacencies, nodes);
        }

        if (initialGraph != null) {
            graph = new EdgeListGraphSingleConnections(initialGraph);
            graph = GraphUtils.replaceNodes(graph, nodes);
        }

        addRequiredEdges(graph);

        if (faithfulnessAssumed) {
            initializeForwardEdgesFromEmptyGraph(getVariables());

            // Do forward search.
            this.mode = Mode.heuristicSpeedup;
            fes();
            bes();

            this.mode = Mode.coverNoncolliders;
            initializeTwoStepEdges(getVariables());
            fes();
            bes();
        } else {
            initializeForwardEdgesFromEmptyGraph(getVariables());

            // Do forward search.
            this.mode = Mode.heuristicSpeedup;
            fes();
            bes();

            this.mode = Mode.allowUnfaithfulness;
            initializeForwardEdgesFromExistingGraph(getVariables());
            fes();
            bes();
        }

//        this.modelScore = scoreDag(SearchGraphUtils.dagFromPattern(graph), true);

        long endTime = System.currentTimeMillis();
        this.elapsedTime = endTime - start;

        if (verbose) {
            this.logger.forceLogMessage("Returning this graph: " + graph);

            this.logger.log("info", "Elapsed time = " + (elapsedTime) / 1000. + " s");
            this.logger.flush();
        }


        return graph;
    }

    /**
     * @return the background knowledge.
     */
    public IKnowledge getKnowledge() {
        return knowledge;
    }

    /**
     * Sets the background knowledge.
     *
     * @param knowledge the knowledge object, specifying forbidden and required
     *                  edges.
     */
    public void setKnowledge(IKnowledge knowledge) {
        if (knowledge == null) {
            throw new NullPointerException();
        }
        this.knowledge = knowledge;
    }

    public long getElapsedTime() {
        return elapsedTime;
    }

    /**
     * If the true graph is set, askterisks will be printed in log output for
     * the true edges.
     */
    public void setTrueGraph(Graph trueGraph) {
        this.trueGraph = trueGraph;
    }

    /**
     * @return the totalScore of the given DAG, up to a constant.
     */
    public double scoreDag(Graph dag) {
        return scoreDag(dag, false);
    }

    /**
     * @return the list of top scoring graphs.
     */
    public LinkedList<ScoredGraph> getTopGraphs() {
        return topGraphs;
    }

    /**
     * @return the initial graph for the search. The search is initialized to
     * this graph and proceeds from there.
     */
    public Graph getInitialGraph() {
        return initialGraph;
    }

    /**
     * Sets the initial graph.
     */
    public void setInitialGraph(Graph initialGraph) {
        initialGraph = GraphUtils.replaceNodes(initialGraph, variables);

        if (initialGraph != null) {
            if (verbose) {
                out.println("Initial graph variables: " + initialGraph.getNodes());
                out.println("Data set variables: " + variables);
            }

            if (!new HashSet<>(initialGraph.getNodes()).equals(new HashSet<>(variables))) {
                throw new IllegalArgumentException("Variables aren't the same.");
            }
        }

        this.initialGraph = initialGraph;
    }

    /**
     * Sets whether verbose output should be produced.
     */
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    /**
     * Sets the output stream that output (except for log output) should be sent
     * to. By detault System.out.
     */
    public void setOut(PrintStream out) {
        this.out = out;
    }

    /**
     * @return the output stream that output (except for log output) should be
     * sent to.
     */
    public PrintStream getOut() {
        return out;
    }

    /**
     * @return the set of preset adjacenies for the algorithm; edges not in this
     * adjacencies graph will not be added.
     */
    public Graph getAdjacencies() {
        return adjacencies;
    }

    /**
     * Sets the set of preset adjacenies for the algorithm; edges not in this
     * adjacencies graph will not be added.
     */
    public void setAdjacencies(Graph adjacencies) {
        this.adjacencies = adjacencies;
    }

    /**
     * A bound on cycle length.
     *
     * @param cycleBound The bound, >= 1, or -1 for unlimited.
     */
    public void setCycleBound(int cycleBound) {
        if (!(cycleBound == -1 || cycleBound >= 1)) {
            throw new IllegalArgumentException("Cycle bound needs to be -1 or >= 1: " + cycleBound);
        }
        this.cycleBound = cycleBound;
    }

    /**
     * If non-null, edges not adjacent in this graph will not be added.
     */
    public void setBoundGraph(Graph boundGraph) {
        this.boundGraph = GraphUtils.replaceNodes(boundGraph, getVariables());
    }

    /**
     * For BIC totalScore, a multiplier on the penalty term. For continuous
     * searches.
     *
     * @deprecated Use the getters on the individual scores instead.
     */
    public double getPenaltyDiscount() {
        if (score instanceof ISemBicScore) {
            return ((ISemBicScore) score).getPenaltyDiscount();
        } else {
            return 2.0;
        }
    }

    /**
     * @deprecated Use the setters on the individual scores instead.
     */
    public void setSamplePrior(double samplePrior) {
        if (score instanceof LocalDiscreteScore) {
            ((LocalDiscreteScore) score).setSamplePrior(samplePrior);
        }
    }

    /**
     * @deprecated Use the setters on the individual scores instead.
     */
    public void setStructurePrior(double expectedNumParents) {
        if (score instanceof LocalDiscreteScore) {
            ((LocalDiscreteScore) score).setStructurePrior(expectedNumParents);
        }
    }

    /**
     * For BIC totalScore, a multiplier on the penalty term. For continuous
     * searches.
     *
     * @deprecated Use the setters on the individual scores instead.
     */
    public void setPenaltyDiscount(double penaltyDiscount) {
        if (score instanceof ISemBicScore) {
            ((ISemBicScore) score).setPenaltyDiscount(penaltyDiscount);
        }
    }

    /**
     * The maximum of parents any nodes can have in output pattern.
     *
     * @return -1 for unlimited.
     */
    public int getMaxDegree() {
        return maxDegree;
    }

    /**
     * The maximum of parents any nodes can have in output pattern.
     *
     * @param maxDegree -1 for unlimited.
     */
    public void setMaxDegree(int maxDegree) {
        if (maxDegree < -1) {
            throw new IllegalArgumentException();
        }
        this.maxDegree = maxDegree;
    }

    public void setSymmetricFirstStep(boolean symmetricFirstStep) {
        this.symmetricFirstStep = symmetricFirstStep;
    }

    public String logEdgeBayesFactorsString(Graph dag) {
        Map<Edge, Double> factors = logEdgeBayesFactors(dag);
        return logBayesPosteriorFactorsString(factors);
    }

    double getModelScore() {
        return modelScore;
    }

    //===========================PRIVATE METHODS========================//
    //Sets the discrete scoring function to use.
    private void setScore(Score totalScore) {
        this.score = totalScore;

        this.variables = new ArrayList<>();

        for (Node node : totalScore.getVariables()) {
            if (node.getNodeType() == NodeType.MEASURED) {
                this.variables.add(node);
            }
        }

        buildIndexing(totalScore.getVariables());

        this.maxDegree = score.getMaxDegree();
    }

    final int[] count = new int[1];

    private int getMinChunk(int n) {
        // The minimum number of operations to do before parallelizing.
        int minChunk = 100;
        return Math.max(n / maxThreads, minChunk);
    }

    class NodeTaskEmptyGraph implements Callable<Boolean> {

        private final int from;
        private final int to;
        private final List<Node> nodes;
        private final Set<Node> emptySet;

        NodeTaskEmptyGraph(int from, int to, List<Node> nodes, Set<Node> emptySet) {
            this.from = from;
            this.to = to;
            this.nodes = nodes;
            this.emptySet = emptySet;
        }

        @Override
        public Boolean call() {
            for (int i = from; i < to; i++) {
                if ((i + 1) % 1000 == 0) {
                    count[0] += 1000;
                    out.println("Initializing effect edges: " + (count[0]));
                }

                Node y = nodes.get(i);
                neighbors.put(y, emptySet);

                for (int j = i + 1; j < nodes.size() && !Thread.currentThread().isInterrupted(); j++) {
                    Node x = nodes.get(j);

                    if (existsKnowledge()) {
                        if (getKnowledge().isForbidden(x.getName(), y.getName()) && getKnowledge().isForbidden(y.getName(), x.getName())) {
                            continue;
                        }

                        if (invalidSetByKnowledge(y, emptySet)) {
                            continue;
                        }
                    }

                    if (adjacencies != null && !adjacencies.isAdjacentTo(x, y)) {
                        continue;
                    }

                    // start: changed by Fattaneh
                    int child = hashIndices.get(y);
                    int parent = hashIndices.get(x);
                    double bump = 0.0, bump2 = 0.0;

                    // if the initial graph graph is empty, proceed as usual
                    if (initialGraph == null){
                        bump = score.localScoreDiff(parent, child);

                    }
                    else{
                        // if x or y has no adjacency in the initial graph, then proceed as if initial graph is empty
                        if (initialGraph.getAdjacentNodes(x).isEmpty() && initialGraph.getAdjacentNodes(y).isEmpty()) {
                            bump = score.localScoreDiff(parent, child);

                        }
                        // if x or y has adjacencies in the initial graph, then that should be considered in scoring
                        else{
                            int[] parentIndicesY;
                            Set<Node> parentsY = new HashSet<>(initialGraph.getParents(y));
                            parentIndicesY = new int[parentsY.size()];
                            int	c = 0;
                            for (Node p : parentsY) {
                                parentIndicesY[c++] = hashIndices.get(p);
                            }

                            bump  = score.localScoreDiff(parent, child, parentIndicesY);

//							if (verbose2){
//								System.out.println("bump: " + bump);
//								System.out.println("bump w/o parents y: " + score.localScoreDiff(parent, child));
//							}
                        }

                    }

                    // computing the bump of an edge from y (child) --> x (parent)
                    if (symmetricFirstStep) {
                        if (initialGraph == null){
                            bump2 = score.localScoreDiff(child, parent);
                        }
                        else{
                            // if x or y has no adjacency, then proceed as an empty initial graph
                            if (initialGraph.getAdjacentNodes(x).isEmpty() && initialGraph.getAdjacentNodes(y).isEmpty()) {
                                bump2 = score.localScoreDiff(child, parent);

                            }
                            else{
                                int[] parentIndicesX;
                                Set<Node> parentsX = new HashSet<>(initialGraph.getParents(x));
                                parentIndicesX = new int[parentsX.size()];
                                int	c = 0;
                                for (Node p : parentsX) {
                                    parentIndicesX[c++] = hashIndices.get(p);
                                }

                                bump2  = score.localScoreDiff(child, parent, parentIndicesX);

//								if (verbose2){
//									System.out.println("bump2: " + bump2);
//									System.out.println("bump2 w/o parents y: " + score.localScoreDiff(child, parent));
//								}
                            }

                        }

						bump = bump > bump2 ? bump : bump2;
                    }

//                    if (symmetricFirstStep) {
//                        double bump2 = score.localScoreDiff(child, parent);
//                        bump = bump > bump2 ? bump : bump2;
//                    }

                    if (boundGraph != null && !boundGraph.isAdjacentTo(x, y)) {
                        continue;
                    }

                    if (bump > 0) {
                        final Edge edge = Edges.undirectedEdge(x, y);
                        effectEdgesGraph.addEdge(edge);
                    }

                    if (bump > 0) {
                        if (initialGraph == null ){
                            addArrow(x, y, emptySet, emptySet, emptySet, bump);

                            if (!symmetricFirstStep){
                                addArrow(y, x, emptySet, emptySet, emptySet, bump2);
                            }

                        }
                        else{
                            if( initialGraph.getAdjacentNodes(x).isEmpty() && initialGraph.getAdjacentNodes(y).isEmpty()){
                                addArrow(x, y, emptySet, emptySet, emptySet, bump);

                                if (!symmetricFirstStep){
                                    addArrow(y, x, emptySet, emptySet, emptySet, bump2);
                                }
                            }
                            else{
//								System.out.println("x: " +  x+ ", y: " + y);
//								System.out.println("sortedArrows before calculateArrowsForward: " +  sortedArrows);
                                calculateArrowsForward(x, y);
//								System.out.println("sortedArrows after calculateArrowsForward: " +  sortedArrows);
                                calculateArrowsForward(y, x);
//								System.out.println("sortedArrows after calculateArrowsForward IN REVERSE : " +  sortedArrows);
                            }
                        }
                    }
                    if (symmetricFirstStep){
                        if (bump2 > 0) {
                            if (initialGraph == null ){
                                addArrow(y, x, emptySet, emptySet, emptySet, bump2);

                            }
                            else{
                                if( initialGraph.getAdjacentNodes(x).isEmpty() && initialGraph.getAdjacentNodes(y).isEmpty()){
                                    addArrow(y, x, emptySet, emptySet, emptySet, bump2);

                                }
                            }
                        }
                    }
                }
            }

            return true;
        }
    }

    private void initializeForwardEdgesFromEmptyGraph(final List<Node> nodes) {
        sortedArrows = new ConcurrentSkipListSet<>();
        lookupArrows = new ConcurrentHashMap<>();
        neighbors = new ConcurrentHashMap<>();
        final Set<Node> emptySet = new HashSet<>();

        long start = System.currentTimeMillis();
        this.effectEdgesGraph = new EdgeListGraphSingleConnections(nodes);

        List<Callable<Boolean>> tasks = new ArrayList<>();

        int numNodesPerTask = Math.max(100, nodes.size() / maxThreads);

        for (int i = 0; i < nodes.size() && !Thread.currentThread().isInterrupted(); i += numNodesPerTask) {
            NodeTaskEmptyGraph task = new NodeTaskEmptyGraph(i, Math.min(nodes.size(), i + numNodesPerTask),
                    nodes, emptySet);
            tasks.add(task);
        }

        pool.invokeAll(tasks);

        long stop = System.currentTimeMillis();

        if (verbose) {
            out.println("Elapsed initializeForwardEdgesFromEmptyGraph = " + (stop - start) + " ms");
        }
    }

    private void initializeTwoStepEdges(final List<Node> nodes) {
        count[0] = 0;

        sortedArrows = new ConcurrentSkipListSet<>();
        lookupArrows = new ConcurrentHashMap<>();
        neighbors = new ConcurrentHashMap<>();

        if (this.effectEdgesGraph == null) {
            this.effectEdgesGraph = new EdgeListGraph(nodes);
        }

        if (initialGraph != null) {
            for (Edge edge : initialGraph.getEdges()) {
                if (!effectEdgesGraph.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
                    effectEdgesGraph.addUndirectedEdge(edge.getNode1(), edge.getNode2());
                }
            }
        }

        final Set<Node> emptySet = new HashSet<>(0);

        class InitializeFromExistingGraphTask extends RecursiveTask<Boolean> {

            private int chunk;
            private int from;
            private int to;

            private InitializeFromExistingGraphTask(int chunk, int from, int to) {
                this.chunk = chunk;
                this.from = from;
                this.to = to;
            }

            @Override
            protected Boolean compute() {
                if (TaskManager.getInstance().isCanceled()) {
                    return false;
                }

                if (to - from <= chunk) {
                    for (int i = from; i < to && !Thread.currentThread().isInterrupted(); i++) {
                        if ((i + 1) % 1000 == 0) {
                            count[0] += 1000;
                            out.println("Initializing effect edges: " + (count[0]));
                        }

                        Node y = nodes.get(i);

                        Set<Node> g = new HashSet<>();

                        for (Node n : graph.getAdjacentNodes(y)) {
                            for (Node m : graph.getAdjacentNodes(n)) {
                                if (Thread.currentThread().isInterrupted()) {
                                    break;
                                }

                                if (m == y) {
                                    continue;
                                }

                                if (graph.isAdjacentTo(y, m)) {
                                    continue;
                                }

                                if (graph.isDefCollider(m, n, y)) {
                                    continue;
                                }

                                g.add(m);
                            }
                        }

                        for (Node x : g) {
                            if (Thread.currentThread().isInterrupted()) {
                                break;
                            }

                            if (x == y) {
                                throw new IllegalArgumentException();
                            }

                            if (existsKnowledge()) {
                                if (getKnowledge().isForbidden(x.getName(), y.getName()) && getKnowledge().isForbidden(y.getName(), x.getName())) {
                                    continue;
                                }

                                if (invalidSetByKnowledge(y, emptySet)) {
                                    continue;
                                }
                            }

                            if (adjacencies != null && !adjacencies.isAdjacentTo(x, y)) {
                                continue;
                            }

                            calculateArrowsForward(x, y);
                        }
                    }

                    return true;
                } else {
                    int mid = (to + from) / 2;

                    InitializeFromExistingGraphTask left = new InitializeFromExistingGraphTask(chunk, from, mid);
                    InitializeFromExistingGraphTask right = new InitializeFromExistingGraphTask(chunk, mid, to);

                    left.fork();
                    right.compute();
                    left.join();

                    return true;
                }
            }
        }

        pool.invoke(new InitializeFromExistingGraphTask(getMinChunk(nodes.size()), 0, nodes.size()));
    }

    private void initializeForwardEdgesFromExistingGraph(final List<Node> nodes) {
        count[0] = 0;

        sortedArrows = new ConcurrentSkipListSet<>();
        lookupArrows = new ConcurrentHashMap<>();
        neighbors = new ConcurrentHashMap<>();

        if (this.effectEdgesGraph == null) {
            this.effectEdgesGraph = new EdgeListGraph(nodes);
        }

        if (initialGraph != null) {
            for (Edge edge : initialGraph.getEdges()) {
                if (Thread.currentThread().isInterrupted()) {
                    break;
                }

                if (!effectEdgesGraph.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
                    effectEdgesGraph.addUndirectedEdge(edge.getNode1(), edge.getNode2());
                }
            }
        }

        final Set<Node> emptySet = new HashSet<>(0);

        class InitializeFromExistingGraphTask extends RecursiveTask<Boolean> {

            private int chunk;
            private int from;
            private int to;

            private InitializeFromExistingGraphTask(int chunk, int from, int to) {
                this.chunk = chunk;
                this.from = from;
                this.to = to;
            }

            @Override
            protected Boolean compute() {
                if (TaskManager.getInstance().isCanceled()) {
                    return false;
                }

                if (to - from <= chunk) {
                    for (int i = from; i < to && !Thread.currentThread().isInterrupted(); i++) {
                        if ((i + 1) % 1000 == 0) {
                            count[0] += 1000;
                            out.println("Initializing effect edges: " + (count[0]));
                        }

                        // We want to recapture the variables that would have been effect edges if paths hadn't
                        // exactly canceled. These are variables X which are d-connected to the target Y where
                        // X--Y was noe identified as an effect edge earlier.
                        Node y = nodes.get(i);
                        Set<Node> D = new HashSet<>(getUnconditionallyDconnectedVars(y, graph));
                        D.remove(y);
                        D.removeAll(effectEdgesGraph.getAdjacentNodes(y));

                        for (Node x : D) {
                            if (Thread.currentThread().isInterrupted()) {
                                break;
                            }

                            if (existsKnowledge()) {
                                if (getKnowledge().isForbidden(x.getName(), y.getName()) && getKnowledge().isForbidden(y.getName(), x.getName())) {
                                    continue;
                                }

                                if (invalidSetByKnowledge(y, emptySet)) {
                                    continue;
                                }
                            }

                            if (adjacencies != null && !adjacencies.isAdjacentTo(x, y)) {
                                continue;
                            }

                            calculateArrowsForward(x, y);
                        }
                    }

                    return true;
                } else {
                    int mid = (to + from) / 2;

                    InitializeFromExistingGraphTask left = new InitializeFromExistingGraphTask(chunk, from, mid);
                    InitializeFromExistingGraphTask right = new InitializeFromExistingGraphTask(chunk, mid, to);

                    left.fork();
                    right.compute();
                    left.join();

                    return true;
                }
            }
        }

        pool.invoke(new InitializeFromExistingGraphTask(getMinChunk(nodes.size()), 0, nodes.size()));
    }

    private void fes() {
        if (verbose) {
            TetradLogger.getInstance().forceLogMessage("** FORWARD EQUIVALENCE SEARCH");
            out.println("** FORWARD EQUIVALENCE SEARCH");
        }

        int maxDegree = this.maxDegree == -1 ? 1000 : this.maxDegree;

        while (!sortedArrows.isEmpty()) {
            Arrow arrow = sortedArrows.first();
            sortedArrows.remove(arrow);

            Node x = arrow.getA();
            Node y = arrow.getB();

            if (graph.isAdjacentTo(x, y)) {
                continue;
            }

            if (graph.getDegree(x) > maxDegree - 1) {
                continue;
            }

            if (graph.getDegree(y) > maxDegree - 1) {
                continue;
            }

            if (!arrow.getNaYX().equals(getNaYX(x, y))) {
                continue;
            }

            if (!new HashSet<>(getTNeighbors(x, y)).equals(arrow.getTNeighbors())) {
                continue;
            }

            if (!validInsert(x, y, arrow.getHOrT(), getNaYX(x, y))) {
                continue;
            }

            boolean inserted = insert(x, y, arrow.getHOrT(), arrow.getBump());

            if (!inserted) {
                continue;
            }

            Set<Node> visited = reapplyOrientation(x, y, null);
            Set<Node> toProcess = new HashSet<>();

            for (Node node : visited) {
                final Set<Node> neighbors1 = getNeighbors(node);
                final Set<Node> storedNeighbors = this.neighbors.get(node);

                if (!(neighbors1.equals(storedNeighbors))) {
                    toProcess.add(node);
                }
            }

            toProcess.add(x);
            toProcess.add(y);

            reevaluateForward(toProcess);
        }
    }

    private Set<Node> getCommonAdjacents(Node x, Node y) {
        Set<Node> adj = new HashSet<>(graph.getAdjacentNodes(x));
        adj.retainAll(graph.getAdjacentNodes(y));
        return adj;
    }

    private void bes() {
        if (verbose) {
            TetradLogger.getInstance().forceLogMessage("** BACKWARD EQUIVALENCE SEARCH");
            out.println("** BACKWARD EQUIVALENCE SEARCH");
        }

        sortedArrows = new ConcurrentSkipListSet<>();
        lookupArrows = new ConcurrentHashMap<>();
        neighbors = new ConcurrentHashMap<>();

        initializeArrowsBackward();

        while (!sortedArrows.isEmpty()) {
            Arrow arrow = sortedArrows.first();
            sortedArrows.remove(arrow);

            Node x = arrow.getA();
            Node y = arrow.getB();

            if (!graph.isAdjacentTo(x, y)) {
                continue;
            }

            Edge edge = graph.getEdge(x, y);

            if (edge.pointsTowards(x)) {
                continue;
            }

            if (!getNaYX(x, y).equals(arrow.getNaYX())) {
                continue;
            }

            if (!validDelete(x, y, arrow.getHOrT(), arrow.getNaYX())) {
                continue;
            }

            boolean deleted = delete(x, y, arrow.getHOrT(), arrow.getBump(), arrow.getNaYX());

            if (!deleted) {
                continue;
            }

            Set<Node> visited = reapplyOrientation(x, y, arrow.getHOrT());

            Set<Node> toProcess = new HashSet<>();

            for (Node node : visited) {
                final Set<Node> neighbors1 = getNeighbors(node);
                final Set<Node> storedNeighbors = this.neighbors.get(node);

                if (!(neighbors1.equals(storedNeighbors))) {
                    toProcess.add(node);
                }
            }

            toProcess.add(x);
            toProcess.add(y);
            toProcess.addAll(getCommonAdjacents(x, y));

            reevaluateBackward(toProcess);
        }
    }

    private Set<Node> reapplyOrientation(Node x, Node y, Set<Node> newArrows) {
        Set<Node> toProcess = new HashSet<>();
        toProcess.add(x);
        toProcess.add(y);

        if (newArrows != null) {
            toProcess.addAll(newArrows);
        }

        return meekOrientRestricted(new ArrayList<>(toProcess), getKnowledge());
    }

    // Returns true if knowledge is not empty.
    private boolean existsKnowledge() {
        return !knowledge.isEmpty();
    }

    // Initiaizes the sorted arrows lists for the backward search.
    private void initializeArrowsBackward() {
        for (Edge edge : graph.getEdges()) {
            Node x = edge.getNode1();
            Node y = edge.getNode2();

            if (existsKnowledge()) {
                if (!getKnowledge().noEdgeRequired(x.getName(), y.getName())) {
                    continue;
                }
            }

            if (edge.pointsTowards(y)) {
                calculateArrowsBackward(x, y);
            } else if (edge.pointsTowards(x)) {
                calculateArrowsBackward(y, x);
            } else {
                calculateArrowsBackward(x, y);
                calculateArrowsBackward(y, x);
            }

            this.neighbors.put(x, getNeighbors(x));
            this.neighbors.put(y, getNeighbors(y));
        }
    }

    // Calcuates new arrows based on changes in the graph for the forward search.
    private void reevaluateForward(final Set<Node> nodes) {
        class AdjTask implements Callable<Boolean> {

            private final List<Node> nodes;
            private int from;
            private int to;

            private AdjTask(List<Node> nodes, int from, int to) {
                this.nodes = nodes;
                this.from = from;
                this.to = to;
            }

            @Override
            public Boolean call() {
                for (int _w = from; _w < to; _w++) {
                    Node x = nodes.get(_w);

                    List<Node> adj;

                    if (mode == Mode.heuristicSpeedup) {
                        adj = effectEdgesGraph.getAdjacentNodes(x);
                    } else if (mode == Mode.coverNoncolliders) {
                        Set<Node> g = new HashSet<>();

                        for (Node n : graph.getAdjacentNodes(x)) {
                            for (Node m : graph.getAdjacentNodes(n)) {
                                if (graph.isAdjacentTo(x, m)) {
                                    continue;
                                }

                                if (graph.isDefCollider(m, n, x)) {
                                    continue;
                                }

                                g.add(m);
                            }
                        }

                        adj = new ArrayList<>(g);
                    } else if (mode == Mode.allowUnfaithfulness) {
                        HashSet<Node> D = new HashSet<>(getUnconditionallyDconnectedVars(x, graph));
                        D.remove(x);
                        adj = new ArrayList<>(D);
                    } else {
                        throw new IllegalStateException();
                    }

                    for (Node w : adj) {
                        if (adjacencies != null && !(adjacencies.isAdjacentTo(w, x))) {
                            continue;
                        }

                        if (w == x) {
                            continue;
                        }

                        if (!graph.isAdjacentTo(w, x)) {
                            calculateArrowsForward(w, x);
                        }
                    }
                }

                return true;
            }
        }

        List<Callable<Boolean>> tasks = new ArrayList<>();

        int numNodesPerTask = Math.max(100, nodes.size() / maxThreads);

        for (int i = 0; i < nodes.size() && !Thread.currentThread().isInterrupted(); i += numNodesPerTask) {
            AdjTask task = new AdjTask(new ArrayList<>(nodes), i, Math.min(nodes.size(), i + numNodesPerTask));
            tasks.add(task);
        }

        pool.invokeAll(tasks);
    }

    // Calculates the new arrows for an a->b edge.
    private void calculateArrowsForward(Node a, Node b) {
        if (mode == Mode.heuristicSpeedup && !effectEdgesGraph.isAdjacentTo(a, b)) {
            return;
        }
        if (adjacencies != null && !adjacencies.isAdjacentTo(a, b)) {
            return;
        }
        this.neighbors.put(b, getNeighbors(b));

        clearArrow(a, b);

        if (a == b) {
            throw new IllegalArgumentException();
        }

        if (existsKnowledge()) {
            if (getKnowledge().isForbidden(a.getName(), b.getName())) {
                return;
            }
        }

        Set<Node> naYX = getNaYX(a, b);
        if (!isClique(naYX)) {
            return;
        }

        List<Node> TNeighbors = getTNeighbors(a, b);

        Set<Set<Node>> previousCliques = new HashSet<>();
        previousCliques.add(new HashSet<>());
        Set<Set<Node>> newCliques = new HashSet<>();

        Set<Node> _T = null;
        double _bump = Double.NEGATIVE_INFINITY;

        FOR:
        for (int i = 0; i <= TNeighbors.size(); i++) {
            final ChoiceGenerator gen = new ChoiceGenerator(TNeighbors.size(), i);
            int[] choice;

            while ((choice = gen.next()) != null) {
                Set<Node> T = GraphUtils.asSet(choice, TNeighbors);

                Set<Node> union = new HashSet<>(naYX);
                union.addAll(T);

                boolean foundAPreviousClique = false;

                for (Set<Node> clique : previousCliques) {
                    if (union.containsAll(clique)) {
                        foundAPreviousClique = true;
                        break;
                    }
                }

                if (!foundAPreviousClique) {
                    break FOR;
                }

                if (!isClique(union)) {
                    continue;
                }
                newCliques.add(union);

                double bump = insertEval(a, b, T, naYX, hashIndices);

                if (bump > 0) {
                    _T = T;
                    _bump = bump;
//                    addArrow(a, b, TNeighbors, naYX, bump);
                }
            }

            if (_bump > Double.NEGATIVE_INFINITY) {
                addArrow(a, b, _T, new HashSet<>(TNeighbors), naYX, _bump);
            }

            previousCliques = newCliques;
            newCliques = new HashSet<>();
        }
    }

    private void addArrow(Node a, Node b, Set<Node> hOrT, Set<Node> TNeighbors, Set<Node> naYX, double bump) {
        Arrow arrow = new Arrow(bump, a, b, hOrT, TNeighbors, naYX, arrowIndex++);
        sortedArrows.add(arrow);
        addLookupArrow(a, b, arrow);
    }

    // Reevaluates arrows after removing an edge from the graph.
    private void reevaluateBackward(Set<Node> toProcess) {
        class BackwardTask extends RecursiveTask<Boolean> {

            private final Node r;
            private List<Node> adj;
            private Map<Node, Integer> hashIndices;
            private int chunk;
            private int from;
            private int to;

            private BackwardTask(Node r, List<Node> adj, int chunk, int from, int to,
                                 Map<Node, Integer> hashIndices) {
                this.adj = adj;
                this.hashIndices = hashIndices;
                this.chunk = chunk;
                this.from = from;
                this.to = to;
                this.r = r;
            }

            @Override
            protected Boolean compute() {
                if (to - from <= chunk) {
                    for (int _w = from; _w < to; _w++) {
                        final Node w = adj.get(_w);
                        Edge e = graph.getEdge(w, r);

                        if (e != null) {
                            if (e.pointsTowards(r)) {
                                calculateArrowsBackward(w, r);
                            } else if (e.pointsTowards(w)) {
                                calculateArrowsBackward(r, w);
                            } else if (Edges.isUndirectedEdge(graph.getEdge(w, r))) {
                                calculateArrowsBackward(w, r);
                                calculateArrowsBackward(r, w);
                            }
                        }
                    }

                    return true;
                } else {
                    int mid = (to - from) / 2;

                    List<BackwardTask> tasks = new ArrayList<>();

                    tasks.add(new BackwardTask(r, adj, chunk, from, from + mid, hashIndices));
                    tasks.add(new BackwardTask(r, adj, chunk, from + mid, to, hashIndices));

                    invokeAll(tasks);

                    return true;
                }
            }
        }

        for (Node r : toProcess) {
            this.neighbors.put(r, getNeighbors(r));
            List<Node> adjacentNodes = graph.getAdjacentNodes(r);
            pool.invoke(new BackwardTask(r, adjacentNodes, getMinChunk(adjacentNodes.size()), 0,
                    adjacentNodes.size(), hashIndices));
        }
    }

    // Calculates the arrows for the removal in the backward direction.
    private void calculateArrowsBackward(Node a, Node b) {
        if (existsKnowledge()) {
            if (!getKnowledge().noEdgeRequired(a.getName(), b.getName())) {
                return;
            }
        }

        clearArrow(a, b);

        Set<Node> naYX = getNaYX(a, b);

        List<Node> _naYX = new ArrayList<>(naYX);

        final int _depth = _naYX.size();

        Set<Node> _h = null;
        double _bump = Double.NEGATIVE_INFINITY;

        final DepthChoiceGenerator gen = new DepthChoiceGenerator(_naYX.size(), _depth);
        int[] choice;

        while ((choice = gen.next()) != null) {
            Set<Node> h = GraphUtils.asSet(choice, _naYX);

            if (existsKnowledge()) {
                if (invalidSetByKnowledge(b, h)) {
                    continue;
                }
            }

            double bump = deleteEval(a, b, h, naYX, hashIndices);

            if (bump >= 0.0) {
                _h = h;
                _bump = bump;
            }
        }

        if (_bump > Double.NEGATIVE_INFINITY) {
            addArrow(a, b, _h, null, naYX, _bump);
        }
    }

    // Basic data structure for an arrow a->b considered for addition or removal from the graph, together with
// associated sets needed to make this determination. For both forward and backward direction, NaYX is needed.
// For the forward direction, TNeighbors neighbors are needed; for the backward direction, H neighbors are needed.
// See Chickering (2002). The totalScore difference resulting from added in the edge (hypothetically) is recorded
// as the "bump".
    private static class Arrow implements Comparable<Arrow> {

        private double bump;
        private Node a;
        private Node b;
        private Set<Node> hOrT;
        private Set<Node> TNeighbors;
        private Set<Node> naYX;
        private int index;

        Arrow(double bump, Node a, Node b, Set<Node> hOrT, Set<Node> capTorH, Set<Node> naYX, int index) {
            this.bump = bump;
            this.a = a;
            this.b = b;
            this.setTNeighbors(capTorH);
            this.hOrT = hOrT;
            this.naYX = naYX;
            this.index = index;
        }

        public double getBump() {
            return bump;
        }

        public Node getA() {
            return a;
        }

        public Node getB() {
            return b;
        }

        Set<Node> getHOrT() {
            return hOrT;
        }

        Set<Node> getNaYX() {
            return naYX;
        }

        // Sorting by bump, high to low. The problem is the SortedSet contains won't add a new element if it compares
        // to zero with an existing element, so for the cases where the comparison is to zero (i.e. have the same
        // bump, we need to determine as quickly as possible a determinate ordering (fixed) ordering for two variables.
        // The fastest way to do this is using a hash code, though it's still possible for two Arrows to have the
        // same hash code but not be equal. If we're paranoid, in this case we calculate a determinate comparison
        // not equal to zero by keeping a list. This last part is commened out by default.
        public int compareTo(Arrow arrow) {
            if (arrow == null) {
                throw new NullPointerException();
            }

            final int compare = Double.compare(arrow.getBump(), getBump());

            if (compare == 0) {
                return Integer.compare(getIndex(), arrow.getIndex());
            }

            return compare;
        }

        public String toString() {
            return "Arrow<" + a + "->" + b + " bump = " + bump + " t/h = " + hOrT + " naYX = " + naYX + ">";
        }

        public int getIndex() {
            return index;
        }

        Set<Node> getTNeighbors() {
            return TNeighbors;
        }

        void setTNeighbors(Set<Node> TNeighbors) {
            this.TNeighbors = TNeighbors;
        }

    }

    // Get all adj that are connected to Y by an undirected edge and not adjacent to X.
    private List<Node> getTNeighbors(Node x, Node y) {
        List<Edge> yEdges = graph.getEdges(y);
        List<Node> tNeighbors = new ArrayList<>();

        for (Edge edge : yEdges) {
            if (!Edges.isUndirectedEdge(edge)) {
                continue;
            }

            Node z = edge.getDistalNode(y);

            if (graph.isAdjacentTo(z, x)) {
                continue;
            }

            tNeighbors.add(z);
        }

        return tNeighbors;
    }

    // Get all adj that are connected to Y.
    private Set<Node> getNeighbors(Node y) {
        List<Edge> yEdges = graph.getEdges(y);
        Set<Node> neighbors = new HashSet<>();

        for (Edge edge : yEdges) {
            if (!Edges.isUndirectedEdge(edge)) {
                continue;
            }

            Node z = edge.getDistalNode(y);

            neighbors.add(z);
        }

        return neighbors;
    }

    // Evaluate the Insert(X, Y, TNeighbors) operator (Definition 12 from Chickering, 2002).
    private double insertEval(Node x, Node y, Set<Node> t, Set<Node> naYX,
                              Map<Node, Integer> hashIndices) {

        if (x == y) {
            throw new IllegalArgumentException();
        }
        Set<Node> set = new HashSet<>(naYX);
        set.addAll(t);
        set.addAll(graph.getParents(y));
        return scoreGraphChange(x, y, set, hashIndices);
    }

    // Evaluate the Delete(X, Y, TNeighbors) operator (Definition 12 from Chickering, 2002).
    private double deleteEval(Node x, Node y, Set<Node> h, Set<Node> naYX,
                              Map<Node, Integer> hashIndices) {
        Set<Node> set = new HashSet<>(naYX);
        set.removeAll(h);
        final List<Node> parents = graph.getParents(y);
        parents.remove(x);
        set.addAll(parents);
        return -scoreGraphChange(x, y, set, hashIndices);
    }

    // Do an actual insertion. (Definition 12 from Chickering, 2002).
    private boolean insert(Node x, Node y, Set<Node> T, double bump) {
        if (graph.isAdjacentTo(x, y)) {
            return false; // The initial graph may already have put this edge in the graph.
        }

        Edge trueEdge = null;

        if (trueGraph != null) {
            Node _x = trueGraph.getNode(x.getName());
            Node _y = trueGraph.getNode(y.getName());
            trueEdge = trueGraph.getEdge(_x, _y);
        }

        if (boundGraph != null && !boundGraph.isAdjacentTo(x, y)) {
            return false;
        }

        graph.addDirectedEdge(x, y);

        if (verbose) {
            String label = trueGraph != null && trueEdge != null ? "*" : "";
            TetradLogger.getInstance().forceLogMessage("graph.getNumEdges()" + ". INSERT " + graph.getEdge(x, y)
                    + " " + T + " " + bump + " " + label);
            out.println(graph.getNumEdges() + ". INSERT " + graph.getEdge(x, y)
                    + " " + T + " " + bump + " " + label);
        }

        int numEdges = graph.getNumEdges();

        if (numEdges % 1000 == 0) {
            out.println("Num edges added: " + numEdges);
        }

        if (verbose) {
            String label = trueGraph != null && trueEdge != null ? "*" : "";
            final String message = graph.getNumEdges() + ". INSERT " + graph.getEdge(x, y)
                    + " " + T + " " + bump + " " + label
                    + " degree = " + GraphUtils.getDegree(graph)
                    + " indegree = " + GraphUtils.getIndegree(graph);
            TetradLogger.getInstance().forceLogMessage(message);
            out.println(message);
        }

        for (Node _t : T) {
            graph.removeEdge(_t, y);
            if (boundGraph != null && !boundGraph.isAdjacentTo(_t, y)) {
                continue;
            }

            graph.addDirectedEdge(_t, y);

            if (verbose) {
                String message = "--- Directing " + graph.getEdge(_t, y);
                TetradLogger.getInstance().forceLogMessage(message);
                out.println(message);
            }
        }

        return true;
    }

    // Do an actual deletion (Definition 13 from Chickering, 2002).
    private boolean delete(Node x, Node y, Set<Node> H, double bump, Set<Node> naYX) {
        Edge trueEdge = null;

        if (trueGraph != null) {
            Node _x = trueGraph.getNode(x.getName());
            Node _y = trueGraph.getNode(y.getName());
            trueEdge = trueGraph.getEdge(_x, _y);
        }

        Edge oldxy = graph.getEdge(x, y);

        Set<Node> diff = new HashSet<>(naYX);
        diff.removeAll(H);

        graph.removeEdge(oldxy);

        int numEdges = graph.getNumEdges();
        if (numEdges % 1000 == 0) {
            out.println("Num edges (backwards) = " + numEdges);
        }

        if (verbose) {
            String label = trueGraph != null && trueEdge != null ? "*" : "";
            String message = (graph.getNumEdges()) + ". DELETE " + x + "-->" + y
                    + " H = " + H + " NaYX = " + naYX + " diff = " + diff + " (" + bump + ") " + label;
            TetradLogger.getInstance().forceLogMessage(message);
            out.println(message);
        }

        for (Node h : H) {
            if (graph.isParentOf(h, y) || graph.isParentOf(h, x)) {
                continue;
            }

            Edge oldyh = graph.getEdge(y, h);

            graph.removeEdge(oldyh);

            graph.addEdge(Edges.directedEdge(y, h));

            if (verbose) {
                TetradLogger.getInstance().forceLogMessage("--- Directing " + oldyh + " to "
                        + graph.getEdge(y, h));
                out.println("--- Directing " + oldyh + " to " + graph.getEdge(y, h));
            }

            Edge oldxh = graph.getEdge(x, h);

            if (Edges.isUndirectedEdge(oldxh)) {
                graph.removeEdge(oldxh);

                graph.addEdge(Edges.directedEdge(x, h));

                if (verbose) {
                    TetradLogger.getInstance().forceLogMessage("--- Directing " + oldxh + " to "
                            + graph.getEdge(x, h));
                    out.println("--- Directing " + oldxh + " to " + graph.getEdge(x, h));
                }
            }
        }

        return true;
    }

    // Test if the candidate insertion is a valid operation
    // (Theorem 15 from Chickering, 2002).
    private boolean validInsert(Node x, Node y, Set<Node> T, Set<Node> naYX) {
        boolean violatesKnowledge = false;

        if (existsKnowledge()) {
            if (knowledge.isForbidden(x.getName(), y.getName())) {
                violatesKnowledge = true;
            }

            for (Node t : T) {
                if (knowledge.isForbidden(t.getName(), y.getName())) {
                    violatesKnowledge = true;
                }
            }
        }

        Set<Node> union = new HashSet<>(T);
        union.addAll(naYX);
        boolean clique = isClique(union);
        boolean noCycle = !existsUnblockedSemiDirectedPath(y, x, union, cycleBound);
        return clique && noCycle && !violatesKnowledge;
    }

    private boolean validDelete(Node x, Node y, Set<Node> H, Set<Node> naYX) {
        boolean violatesKnowledge = false;

        if (existsKnowledge()) {
            for (Node h : H) {
                if (knowledge.isForbidden(x.getName(), h.getName())) {
                    violatesKnowledge = true;
                }

                if (knowledge.isForbidden(y.getName(), h.getName())) {
                    violatesKnowledge = true;
                }
            }
        }

        Set<Node> diff = new HashSet<>(naYX);
        diff.removeAll(H);
        return isClique(diff) && !violatesKnowledge;
    }

    // Adds edges required by knowledge.
    private void addRequiredEdges(Graph graph) {
        if (!existsKnowledge()) {
            return;
        }

        for (Iterator<KnowledgeEdge> it = getKnowledge().requiredEdgesIterator(); it.hasNext() && !Thread.currentThread().isInterrupted(); ) {
            KnowledgeEdge next = it.next();

            Node nodeA = graph.getNode(next.getFrom());
            Node nodeB = graph.getNode(next.getTo());

            if (!graph.isAncestorOf(nodeB, nodeA)) {
                graph.removeEdges(nodeA, nodeB);
                graph.addDirectedEdge(nodeA, nodeB);

                if (verbose) {
                    TetradLogger.getInstance().forceLogMessage("Adding edge by knowledge: " + graph.getEdge(nodeA, nodeB));
                    out.println("Adding edge by knowledge: " + graph.getEdge(nodeA, nodeB));
                }
            }
        }
        for (Edge edge : graph.getEdges()) {
            if (Thread.currentThread().isInterrupted()) {
                break;
            }

            final String A = edge.getNode1().getName();
            final String B = edge.getNode2().getName();

            if (knowledge.isForbidden(A, B)) {
                Node nodeA = edge.getNode1();
                Node nodeB = edge.getNode2();
                if (nodeA == null || nodeB == null) {
                    throw new NullPointerException();
                }

                if (graph.isAdjacentTo(nodeA, nodeB) && !graph.isChildOf(nodeA, nodeB)) {
                    if (!graph.isAncestorOf(nodeA, nodeB)) {
                        graph.removeEdges(nodeA, nodeB);
                        graph.addDirectedEdge(nodeB, nodeA);

                        if (verbose) {
                            TetradLogger.getInstance().forceLogMessage("Adding edge by knowledge: " + graph.getEdge(nodeB, nodeA));
                            out.println("Adding edge by knowledge: " + graph.getEdge(nodeB, nodeA));
                        }
                    }
                }

                if (!graph.isChildOf(nodeA, nodeB) && getKnowledge().isForbidden(nodeA.getName(), nodeB.getName())) {
                    if (!graph.isAncestorOf(nodeA, nodeB)) {
                        graph.removeEdges(nodeA, nodeB);
                        graph.addDirectedEdge(nodeB, nodeA);

                        if (verbose) {
                            TetradLogger.getInstance().forceLogMessage("Adding edge by knowledge: " + graph.getEdge(nodeB, nodeA));
                            out.println("Adding edge by knowledge: " + graph.getEdge(nodeB, nodeA));
                        }
                    }
                }
            } else if (knowledge.isForbidden(B, A)) {
                Node nodeA = edge.getNode2();
                Node nodeB = edge.getNode1();
                if (nodeA == null || nodeB == null) {
                    throw new NullPointerException();
                }

                if (graph.isAdjacentTo(nodeA, nodeB) && !graph.isChildOf(nodeA, nodeB)) {
                    if (!graph.isAncestorOf(nodeA, nodeB)) {
                        graph.removeEdges(nodeA, nodeB);
                        graph.addDirectedEdge(nodeB, nodeA);

                        if (verbose) {
                            TetradLogger.getInstance().forceLogMessage("Adding edge by knowledge: " + graph.getEdge(nodeB, nodeA));
                            out.println("Adding edge by knowledge: " + graph.getEdge(nodeB, nodeA));
                        }
                    }
                }
                if (!graph.isChildOf(nodeA, nodeB) && getKnowledge().isForbidden(nodeA.getName(), nodeB.getName())) {
                    if (!graph.isAncestorOf(nodeA, nodeB)) {
                        graph.removeEdges(nodeA, nodeB);
                        graph.addDirectedEdge(nodeB, nodeA);

                        if (verbose) {
                            TetradLogger.getInstance().forceLogMessage("Adding edge by knowledge: " + graph.getEdge(nodeB, nodeA));
                            out.println("Adding edge by knowledge: " + graph.getEdge(nodeB, nodeA));
                        }
                    }
                }
            }
        }
    }

    // Use background knowledge to decide if an insert or delete operation does not orient edges in a forbidden
    // direction according to prior knowledge. If some orientation is forbidden in the subset, the whole subset is
    // forbidden.
    private boolean invalidSetByKnowledge(Node y, Set<Node> subset) {
        for (Node node : subset) {
            if (getKnowledge().isForbidden(node.getName(), y.getName())) {
                return true;
            }
        }
        return false;
    }

    // Find all adj that are connected to Y by an undirected edge that are adjacent to X (that is, by undirected or
    // directed edge).
    private Set<Node> getNaYX(Node x, Node y) {
        List<Node> adj = graph.getAdjacentNodes(y);
        Set<Node> nayx = new HashSet<>();

        for (Node z : adj) {
            if (z == x) {
                continue;
            }
            Edge yz = graph.getEdge(y, z);
            if (!Edges.isUndirectedEdge(yz)) {
                continue;
            }
            if (!graph.isAdjacentTo(z, x)) {
                continue;
            }
            nayx.add(z);
        }

        return nayx;
    }

    // Returns true iif the given set forms a clique in the given graph.
    private boolean isClique(Set<Node> nodes) {
        List<Node> _nodes = new ArrayList<>(nodes);
        for (int i = 0; i < _nodes.size(); i++) {
            for (int j = i + 1; j < _nodes.size(); j++) {
                if (!graph.isAdjacentTo(_nodes.get(i), _nodes.get(j))) {
                    return false;
                }
            }
        }

        return true;
    }

    // Returns true if a path consisting of undirected and directed edges toward 'to' exists of
    // length at most 'bound'. Cycle checker in other words.
    private boolean existsUnblockedSemiDirectedPath(Node from, Node to, Set<Node> cond, int bound) {
        Queue<Node> Q = new LinkedList<>();
        Set<Node> V = new HashSet<>();
        Q.offer(from);
        V.add(from);
        Node e = null;
        int distance = 0;

        while (!Q.isEmpty()) {
            Node t = Q.remove();
            if (t == to) {
                return true;
            }

            if (e == t) {
                e = null;
                distance++;
                if (distance > (bound == -1 ? 1000 : bound)) {
                    return false;
                }
            }

            for (Node u : graph.getAdjacentNodes(t)) {
                Edge edge = graph.getEdge(t, u);
                Node c = traverseSemiDirected(t, edge);
                if (c == null) {
                    continue;
                }
                if (cond.contains(c)) {
                    continue;
                }

                if (c == to) {
                    return true;
                }

                if (!V.contains(c)) {
                    V.add(c);
                    Q.offer(c);

                    if (e == null) {
                        e = u;
                    }
                }
            }
        }

        return false;
    }

    // Used to find semidirected paths for cycle checking.
    private static Node traverseSemiDirected(Node node, Edge edge) {
        if (node == edge.getNode1()) {
            if (edge.getEndpoint1() == Endpoint.TAIL) {
                return edge.getNode2();
            }
        } else if (node == edge.getNode2()) {
            if (edge.getEndpoint2() == Endpoint.TAIL) {
                return edge.getNode1();
            }
        }
        return null;
    }

    // Runs Meek rules on just the changed adj.
    private Set<Node> meekOrientRestricted(List<Node> nodes, IKnowledge knowledge) {
        MeekRules rules = new MeekRules();
        rules.setKnowledge(knowledge);
        rules.setUndirectUnforcedEdges(true);
        rules.orientImplied(graph, nodes);
        return rules.getVisited();
    }

    // Maps adj to their indices for quick lookup.
    private void buildIndexing(List<Node> nodes) {
        this.hashIndices = new ConcurrentHashMap<>();

        int i = -1;

        for (Node n : nodes) {
            this.hashIndices.put(n, ++i);
        }
    }

    // Removes information associated with an edge x->y.
    private synchronized void clearArrow(Node x, Node y) {
        final OrderedPair<Node> pair = new OrderedPair<>(x, y);
        final Set<Arrow> lookupArrows = this.lookupArrows.get(pair);

        if (lookupArrows != null) {
            sortedArrows.removeAll(lookupArrows);
        }

        this.lookupArrows.remove(pair);
    }

    // Adds the given arrow for the adjacency i->j. These all are for i->j but may have
    // different TNeighbors or H or NaYX sets, and so different bumps.
    private void addLookupArrow(Node i, Node j, Arrow arrow) {
        OrderedPair<Node> pair = new OrderedPair<>(i, j);
        Set<Arrow> arrows = lookupArrows.get(pair);

        if (arrows == null) {
            arrows = new ConcurrentSkipListSet<>();
            lookupArrows.put(pair, arrows);
        }

        arrows.add(arrow);
    }

    //===========================SCORING METHODS===================//

    private double scoreDag(Graph dag, boolean recordScores) {

//        if (score instanceof GraphScore) return 0.0;

        Score score2 = score;

        if (score instanceof SemBicScore) {

            DataSet dataSet = ((SemBicScore) score).getDataSet();

            if (dataSet != null) {
                score2 = new SemBicScore(dataSet);
            } else {
                ICovarianceMatrix cov = ((SemBicScore) score).getCovariances();

                if (cov != null) {
                    score2 = new SemBicScore(cov);
                }
            }
        }

        dag = GraphUtils.replaceNodes(dag, getVariables());

        if (dag == null) throw new NullPointerException("DAG not specified.");

        double _score = 0;

        for (Node node : getVariables()) {

            if (score2 instanceof BDeuScore) {
                List<Node> x = dag.getParents(node);

                int[] parentIndices = new int[x.size()];

                int count = 0;
//    			System.out.println(node);
                for (Node parent : x) {
                    parentIndices[count++] = hashIndices.get(parent);
                }
//                final double bic = (score2.localScore1(hashIndices.get(node), parentIndices);
                final double bic =  score.localScore(hashIndices.get(x), parentIndices);
//    			System.out.println( "node " + node +", pa (" + node + ") = " + Arrays.toString(parentIndices) +" =" + bic);

                if (recordScores) {
                    node.addAttribute("BIC", bic);
                }

                _score += bic;
            }
        }

        if (recordScores) {
            graph.addAttribute("BIC", _score);
        }

        return _score;
    }

    private double scoreGraphChange(Node x, Node y, Set<Node> parents,
                                    Map<Node, Integer> hashIndices) {
        int yIndex = hashIndices.get(y);

        if (x == y) {
            throw new IllegalArgumentException();
        }
        if (parents.contains(y)) {
            throw new IllegalArgumentException();
        }

        int[] parentIndices = new int[parents.size()];

        int count = 0;
        for (Node parent : parents) {
            parentIndices[count++] = hashIndices.get(parent);
        }

        return score.localScoreDiff(hashIndices.get(x), yIndex, parentIndices);
    }

    private List<Node> getVariables() {
        return variables;
    }

    private Map<Edge, Double> logEdgeBayesFactors(Graph dag) {
        Map<Edge, Double> logBayesFactors = new HashMap<>();
        double withEdge = scoreDag(dag);

        for (Edge edge : dag.getEdges()) {
            dag.removeEdge(edge);
            double withoutEdge = scoreDag(dag);
            double difference = withEdge - withoutEdge;
            logBayesFactors.put(edge, difference);
            dag.addEdge(edge);
        }

        return logBayesFactors;
    }

    private String logBayesPosteriorFactorsString(final Map<Edge, Double> factors) {
        NumberFormat nf = new DecimalFormat("0.00");
        StringBuilder builder = new StringBuilder();

        List<Edge> edges = new ArrayList<>(factors.keySet());

        edges.sort((o1, o2) -> -Double.compare(factors.get(o1), factors.get(o2)));

        builder.append("Edge Posterior Log Bayes Factors:\n\n");

        builder.append("For a DAG in the IMaGES pattern with model totalScore m, for each edge e in the "
                + "DAG, the model totalScore that would result from removing each edge, calculating "
                + "the resulting model totalScore m(e), and then reporting m - m(e). The totalScore used is "
                + "the IMScore, L - SUM_i{kc ln n(i)}, L is the maximum likelihood of the model, "
                + "k isthe number of parameters of the model, n(i) is the sample size of the ith "
                + "data set, and c is the penalty penaltyDiscount. Note that the more negative the totalScore, "
                + "the more important the edge is to the posterior probability of the IMaGES model. "
                + "Edges are given in order of their importance so measured.\n\n");

        int i = 0;

        for (Edge edge : edges) {
            builder.append(++i).append(". ").append(edge).append(" ").append(nf.format(factors.get(edge))).append("\n");
        }

        return builder.toString();
    }

    // Only need the unconditioal d-connection here.
    private static Set<Node> getUnconditionallyDconnectedVars(Node x, Graph graph) {
        Set<Node> Y = new HashSet<>();

        class EdgeNode {

            private Edge edge;
            private Node node;

            private EdgeNode(Edge edge, Node node) {
                this.edge = edge;
                this.node = node;
            }

            public int hashCode() {
                return edge.hashCode() + node.hashCode();
            }

            public boolean equals(Object o) {
                if (!(o instanceof EdgeNode)) {
                    throw new IllegalArgumentException();
                }
                EdgeNode _o = (EdgeNode) o;
                return _o.edge == edge && _o.node == node;
            }
        }

        Queue<EdgeNode> Q = new ArrayDeque<>();
        Set<EdgeNode> V = new HashSet<>();

        for (Edge edge : graph.getEdges(x)) {
            EdgeNode edgeNode = new EdgeNode(edge, x);
            Q.offer(edgeNode);
            V.add(edgeNode);
            Y.add(edge.getDistalNode(x));
        }

        while (!Q.isEmpty()) {
            EdgeNode t = Q.poll();

            Edge edge1 = t.edge;
            Node a = t.node;
            Node b = edge1.getDistalNode(a);

            for (Edge edge2 : graph.getEdges(b)) {
                Node c = edge2.getDistalNode(b);
                if (c == a) {
                    continue;
                }

                if (reachable(edge1, edge2, a)) {
                    EdgeNode u = new EdgeNode(edge2, b);

                    if (!V.contains(u)) {
                        V.add(u);
                        Q.offer(u);
                        Y.add(c);
                    }
                }
            }
        }

        return Y;
    }

    private static boolean reachable(Edge e1, Edge e2, Node a) {
        Node b = e1.getDistalNode(a);

        boolean collider = e1.getProximalEndpoint(b) == Endpoint.ARROW
                && e2.getProximalEndpoint(b) == Endpoint.ARROW;

        return !collider;
    }
}
