package edu.cmu.tetrad.algcomparison.independence;

import edu.cmu.tetrad.annotation.TestOfIndependence;
import edu.cmu.tetrad.data.DataModel;
import edu.cmu.tetrad.data.DataType;
import edu.cmu.tetrad.data.DataUtils;
import edu.cmu.tetrad.search.IndTestConditionalGaussianLRT;
import edu.cmu.tetrad.search.IndTestRBIT;
import edu.cmu.tetrad.search.IndependenceTest;
import edu.cmu.tetrad.util.Parameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Wrapper for Random Bayesian Independence Test.
 *
 * @author Bryan Andrews
 */
@TestOfIndependence(
        name = "Random Bayesian Independence Test",
        command = "rbit",
        dataType = DataType.Continuous
)
public class RBIT implements IndependenceWrapper {

    static final long serialVersionUID = 23L;

    @Override
    public IndependenceTest getTest(DataModel dataSet, Parameters parameters) {
        final IndTestRBIT test = new IndTestRBIT(DataUtils.getMixedDataSet(dataSet));
        return test;
    }

    @Override
    public String getDescription() {
        return "Random Bayesian Independence Test";
    }

    @Override
    public DataType getDataType() {
        return DataType.Continuous;
    }

    @Override
    public List<String> getParameters() {
        List<String> parameters = new ArrayList<>();
        return parameters;
    }

}
