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
        double temp,sum=0;

        /*for (Instance current : getLabelledData().getDataset()) {
            temp = 1 / (1 + euc.distance(instance, current));
            if(temp > max)
                max = temp;
        }
        */

        for (Instance current : getLabelledData().getDataset()) {
            temp = 1 / (1 + euc.distance(instance, current));
            if (temp > max)
                max = temp;
            sum += temp;
        }

        double simUnknown = 1 - max;

        double prob = simUnknown / sum;

        /*
        double sumatoria = 0;
        double log;

        for (double current : probabilities) {

            if (current != 0) {
                log = logbase2(current);
                sumatoria += current * log;
            }
        }

        return (sumatoria != 0) ? (-sumatoria) : 0;
        */
        return 0;
    }

    private double logbase2(double d) {
        return Math.log(d) / Math.log(2);
    }
}
