package net.sf.jclal.activelearning.singlelabel.querystrategy;

import weka.core.Instance;
import weka.core.EuclideanDistance;

public class WithUnknownClassQueryStrategy extends AbstractSingleLabelQueryStrategy {

    public WithUnknownClassQueryStrategy() {
        setMaximal(true);
    }

    @Override
    public double utilityInstance(Instance instance) {
        EuclideanDistance euc = new EuclideanDistance(getLabelledData().getDataset());

        double max = 0;
        double temp;

        for (Instance current : getLabelledData().getDataset()) {
            temp = 1 / (1 + euc.distance(instance, current));
            if(temp > max)
                max = temp;
        }

        double simUnknown = 1 - max;

        return 0;
    }
}
