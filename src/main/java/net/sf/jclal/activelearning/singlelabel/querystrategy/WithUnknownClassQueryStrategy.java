package net.sf.jclal.activelearning.singlelabel.querystrategy;

import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

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
        /* Calculate Probabilities */
        for (i = 0; i < prob.length; i++) {
            prob[i] = prob[i] / sumSim;
        }
        /* Determine Interest */
        double interest = tanh(2*prob[i]);
        /* Determinar Entropy */
        double entropy = 0;
        for (i = 0; i < prob.length; i++) {
            entropy -= prob[i]*log(prob[i]);
        }
        /* Pick highest from interest and entropy and return it*/
        return interest >= entropy ? interest : entropy;
    }
}
