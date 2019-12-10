package net.sf.jclal.activelearning.stopcriteria;

import net.sf.jclal.activelearning.oracle.AbstractOracle;
import net.sf.jclal.core.IAlgorithm;
import net.sf.jclal.core.IConfigure;
import net.sf.jclal.core.IStopCriterion;
import org.apache.commons.configuration.Configuration;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Arrays;

public class SurpriseBased implements IStopCriterion, IConfigure {

    private static final long serialVersionUID = 8148193972540461736L;
    /**
     * Minimum for surprise, by default it is equal to 0.05
     */
    private double maxSurprise = 0.05;
    /**
     * Length of the interval of convergence, by default it is equal to 5
     */
    private int inrConverge = 5;
    /**
     * Surprise values for each interval
     */
    private double[] inrValues;

    @Override
    public boolean stop(IAlgorithm algorithm) {
        int i;
        double maxProb, eventProb, surprise, maxValue;
        int eventProbInd;
        ArrayList<String> queriedInstances;
        int numQueriedInstances, numLabelledInstances;
        Instances labelledInstances;
        Instances oldInstances;
        Instances lastInstances;
        double[] prob;
        // Get Instances Queried To The Oracle In String Format
        queriedInstances = ((AbstractOracle) algorithm.getScenario().getOracle()).getLastLabeledInstances();
        if (queriedInstances != null) {
            // Get Number Of Instances Queried To The Oracle
            numQueriedInstances = queriedInstances.size();
            // Get Instances Labelled By The Oracle
            labelledInstances = algorithm.getLabeledDataSet().getDataset();
            // Get Number Of Instances Labelled By The Oracle
            numLabelledInstances = labelledInstances.numInstances();
            // Get Instances Previously Labelled By The Oracle
            oldInstances = new Instances(labelledInstances, 0, numLabelledInstances - numQueriedInstances);
            // Get Last Instances Labelled By The Oracle
            lastInstances = new Instances(labelledInstances, numLabelledInstances - numQueriedInstances, numQueriedInstances);
            /*
            // Print Instances Labelled By The Oracle
            System.out.println(numLabelledInstances);
            System.out.println(labelledInstances);
            // Print Instances Previously Labelled By The Oracle
            System.out.println(numLabelledInstances - numQueriedInstances);
            System.out.println(oldInstances);
            // Print Last Instances Labelled By The Oracle
            System.out.println(numQueriedInstances);
            System.out.println(lastInstances);
             */
            for (i = 0; i < numQueriedInstances; i++) {
                // Get Probability Distribution
                prob = getProbabilityDistribution(oldInstances, lastInstances.get(i));
                /*
                // Print Probability Distribution
                System.out.println(Arrays.toString(prob));
                 */
                // Get Index For The Probability Of The Class Of The Most Recent Labelled Instance
                eventProbInd = eventProb(oldInstances, lastInstances.get(i));
                // If Class Of The Most Recent Labelled Instance Is New
                if(eventProbInd < 0)
                    surprise = 1.0;
                else{
                    // Get Highest Probability of the Probability Distribution 'prob'
                    maxProb = maxArray(prob);
                    // Get The Probability Of The Class Of The Most Recent Labelled Instance
                    eventProb = prob[eventProbInd];
                    // Calculate Surprise
                    surprise = logbase2(1.0 + maxProb - eventProb);
                    /*
                    // Print Highest Probability of the Probability Distribution
                    System.out.println(maxProb);
                    // Print Probability Of The Class Of The Most Recent Labelled Instance
                    System.out.println(eventProb);
                    // Print Surprise
                    System.out.println(surprise);
                     */
                }
                // Consider Surprise Value In The Interval Window
                shiftInrWindowLeft(surprise);
                // Get Maximum Value In The Interval Window
                maxValue = maxArray(this.inrValues);
                // Stop Training If Certain Criterion Are Valid (Less Or Equal Than The Maximum Surprise)
                if (maxValue < this.maxSurprise)
                    return true;
            }
        }
        return false;
    }

    @Override
    public void configure(Configuration settings) {
        throw new UnsupportedOperationException();
    }

    /**
     * Set the minimum for surprise
     *
     * @param maxSurprise
     *            Set the minimum surprise
     */
    public void setMinSurprise(double maxSurprise) {
        this.maxSurprise = maxSurprise;
    }

    /**
     * Set the length of the interval of convergence
     *
     * @param inrConverge
     *            Set the interval of convergence
     */
    public void setInrConverge(int inrConverge) {
        this.inrConverge = inrConverge;
        this.inrValues = new double[this.inrConverge];
        Arrays.fill(this.inrValues, 1.0);
    }

    private void shiftInrWindowLeft(double newProb) {
        int i;
        for(i = 0; i < this.inrConverge-1 ; i++) {
            this.inrValues[i] = this.inrValues[i+1];
        }
        this.inrValues[i] = newProb;
    }

    private double logbase2(double d) {
        return Math.log(d) / Math.log(2);
    }

    private static int eventProb(Instances labelledInstances, Instance instance) {
        int i, j, ni;
        int ind = 0;
        int numClasses = labelledInstances.numClasses();
        int numInstances =  labelledInstances.numInstances();
        for(i = 0; i < numClasses; i++) {
            ni = 0;
            for (j = 0; j < numInstances; j++) {
                if ((int) labelledInstances.get(j).classValue() == i) {
                    ni++;
                }
            }
            if(ni > 0) {
                if(instance.classValue() != i) {
                    ind++;
                }
                else {
                    return ind;
                }
            }
        }
        return -1;
    }

    private static double maxArray(double[] array) {
        int i;
        double max = 0.0;
        for(i = 0; i < array.length-1; i++) {
            if(array[i] > max)
                max = array[i];
        }
        return max;
    }

    private double[] getProbabilityDistribution(Instances labelledInstances, Instance instance) {
        int i;
        double sim, sumSim, maxSim, unkSim;
        Instances centroids;
        Instance formattedInstance;
        EuclideanDistance euc;
        double[] prob;
        sumSim = 0;
        maxSim = 0;
        // Centroids Calculation
        centroids = getCentroids(labelledInstances);
        // Similarities Calculation
        formattedInstance = getClassless(instance, labelledInstances);
        euc = new EuclideanDistance(centroids);
        euc.setDontNormalize(true);
        prob = new double[centroids.numInstances() + 1];
        for (i = 0; i < centroids.numInstances(); i++) {
            sim = 1 / (1 + euc.distance(formattedInstance, centroids.get(i)));
            prob[i] = sim;
            sumSim += sim;
            if (sim > maxSim)
                maxSim = sim;
        }
        // Similarity of Unknown Calculation
        unkSim = 1 - maxSim;
        prob[i] = unkSim;
        sumSim += unkSim;
        // Calculate Probability Distribution
        for (i = 0; i < prob.length; i++) {
            prob[i] = prob[i] / sumSim;
        }
        // Return Probability Distribution
        return prob;
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
