package net.sf.jclal.activelearning.singlelabel.querystrategy;

import weka.core.Instance;
import weka.core.EuclideanDistance;
import weka.core.Instances;

import static java.lang.Math.log;
import static java.lang.Math.tanh;

public class WithUnknownClassQueryStrategy extends AbstractSingleLabelQueryStrategy {

    public WithUnknownClassQueryStrategy() {
        setMaximal(true);
    }

    @Override
    public double utilityInstance(Instance instance) {
        int i;
        double sim, sumSim, maxSim, unkSim;
        Instances centroids = null;
        EuclideanDistance euc = new EuclideanDistance(getLabelledData().getDataset());
        double[] prob = new double[getLabelledData().getDataset().numClasses() + 1];
        sumSim = 0;
        maxSim = 0;
        /* Centroids Calculation */
        System.out.println(getLabelledData().getDataset());
        System.out.println(getLabelledData().getDataset().firstInstance().classValue());
        try {
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        /* Similarities Calculation */
        for (i = 0; i < centroids.numInstances(); i++) {
            sim = 1 / (1 + euc.distance(instance, centroids.get(i)));
            prob[i] = sim;
            sumSim += sim;
            if (sim > maxSim)
                maxSim = sim;
        }
        /* Similarity of Unknown Calculation */
        unkSim = 1 - maxSim;
        prob[i] = unkSim;
        sumSim += unkSim;
        /* Calculate Probability Distribution */
        for (i = 0; i < prob.length; i++) {
            prob[i] = prob[i] / sumSim;
        }
        /* Determine Unknown Interest */
        double unknownInt = tanh(2 * prob[i-1]);
        /* Determine Uncertainty Interest */
        double uncertainInt = 0;
        for (i = 0; i < prob.length; i++) {
            if(prob[i] != 0)
                uncertainInt -= prob[i] * logbase2(prob[i]);
        }
        /* Pick highest between Unknown Interest and Uncertainty Interest and return it */
        return Math.max(unknownInt, uncertainInt);
    }

    private double logbase2(double d) {
        return Math.log(d) / Math.log(2);
    }
}
