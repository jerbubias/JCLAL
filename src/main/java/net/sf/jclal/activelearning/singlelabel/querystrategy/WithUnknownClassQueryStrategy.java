package net.sf.jclal.activelearning.singlelabel.querystrategy;

import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class WithUnknownClassQueryStrategy extends AbstractSingleLabelQueryStrategy {

    public WithUnknownClassQueryStrategy() {
        setMaximal(true);
    }

    @Override
    public double utilityInstance(Instance instance) {
        int i;
        double sim, sumSim, maxSim, unkSim;
        Instances centroids;
        Instance formattedInstance;
        EuclideanDistance euc;
        double[] prob;
        sumSim = 0;
        maxSim = 0;
        /* Centroids Calculation */
        centroids = getCentroids(getLabelledData().getDataset());
        /* Similarities Calculation */
        formattedInstance = getClassless(instance, getLabelledData().getDataset());
        euc = new EuclideanDistance(centroids);
        prob = new double[centroids.numInstances() + 1];
        for (i = 0; i < centroids.numInstances(); i++) {
            System.out.println(formattedInstance);
            System.out.println(centroids.get(i));
            System.out.println(euc.distance(formattedInstance, centroids.get(i)));
            sim = 1 / (1 + euc.distance(formattedInstance, centroids.get(i)));
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
        double unknownInt = Math.tanh(2 * prob[i-1]);
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

    private Instances getCentroids(Instances instances) {
        int i, j, ni;
        Instances centroids = null;
        int numClasses = instances.numClasses();
        int numInstances =  instances.numInstances();
        for(i = 0; i < numClasses; i++) {
            Instances classInstances = new Instances(instances, 0);
            for(j = 0; j < numInstances; j++) {
                if((int)instances.get(j).classValue() == i) {
                    classInstances.add(instances.get(j));
                }
            }
            ni = classInstances.numInstances();
            if(ni > 0) {
                classInstances = getClassless(classInstances);
                if(centroids == null) {
                    centroids = new Instances(classInstances, 0);
                }
                if(ni == 1) {
                    centroids.add(classInstances.get(0));
                }
                else {
                    SimpleKMeans kMeans = new SimpleKMeans();
                    try {
                        kMeans.setNumClusters(1);
                        kMeans.buildClusterer(classInstances);
                    } catch (Exception e) {
                        e.printStackTrace();
                        continue;
                    }
                    classInstances = kMeans.getClusterCentroids();
                    centroids.add(classInstances.get(0));
                }
            }
        }
        if(centroids == null) {
            centroids = new Instances(instances, 0);
        }
        return centroids;
    }

    private Instance getClassless(Instance instance, Instances instances) {
        Remove filter = new Remove();
        filter.setAttributeIndices("" + (instances.classIndex() + 1));
        try {
            filter.setInputFormat(instances);
            filter.input(instance);
            return filter.output();
        } catch (Exception e) {
            e.printStackTrace();
            return instance;
        }
    }

    private Instances getClassless(Instances instances) {
        Remove filter = new Remove();
        filter.setAttributeIndices("" + (instances.classIndex() + 1));
        try {
            filter.setInputFormat(instances);
            return Filter.useFilter(instances, filter);
        } catch (Exception e) {
            e.printStackTrace();
            return instances;
        }
    }
}
