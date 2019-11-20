package net.sf.jclal.activelearning.stopcriteria;

import net.sf.jclal.activelearning.oracle.AbstractOracle;
import net.sf.jclal.core.IAlgorithm;
import net.sf.jclal.core.IConfigure;
import net.sf.jclal.core.IStopCriterion;
import org.apache.commons.configuration.Configuration;

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
        double maxProb, surprise, maxValue;
        ArrayList<String> lastInstances;
        double[] prob;
        // Get Instances Queried To The Oracle
        lastInstances = ((AbstractOracle) algorithm.getScenario().getOracle()).getLastLabeledInstances();
        if (lastInstances != null) {
            for (i = 0; i < lastInstances.size(); i++) {
                // Get Probability Distribution
                prob = new double[]{0.25, 0.25, 0.25, 0.25};
                maxProb = maxArray(prob);
                // Calculate Surprise
                surprise = 1 - maxProb;
                // Consider Surprise Value In The Interval Window
                shiftInrWindowLeft(surprise);
                // Get Maximum Value In The Interval Window
                maxValue = maxArray(this.inrValues);
                // Stop Training If Certain Criterion Are Valid (Less Or Equal Than The Maximum Surprise)
                if (maxValue > this.maxSurprise)
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

    private static double maxArray(double[] array) {
        int i;
        double max = 0.0;
        for(i = 0; i < array.length; i++) {
            if(array[i] > max)
                max = array[i];
        }
        return max;
    }
}
